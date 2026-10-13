"""The coordinator: control plane of a distributed training run over a WAN.

It never touches gradient numerics; boundary tensors flow worker to worker.
What it owns:

  registry      worker agents dial in, authenticate, and register what they
                measured about themselves (resources/probe.py); a worker that
                loses its connection and dials back is re-attached, not new;
  selection     which registered workers form the fleet (WAN-aware, link
                memory first, heterogeneous by default); the rest are spares;
  routing       the pairwise link matrix, the chain order that maximises its
                worst hop, and a full measurement of every chosen hop with
                re-routing when a hop measures far below its probe;
  planning      recipe-given placement (placement.py) or full planning from a
                job description (fleetplan.py), plus an on-device memory fit
                check that re-plans at a smaller microbatch when a stage does
                not fit;
  launch        one pinned spec to every slot, last rank first;
  monitoring    heartbeats, per-step progress, exits (fault_tolerance/liveness);
  checkpoints   every wave every slot writes is shipped here and verified, so
                a dead worker does not take its shard with it;
  recovery      a convicted slot is replaced from the spares and every slot
                resumes from the newest common wave under the original spec;
                a fleet that is alive but cannot progress (a broken WAN path)
                is restarted in place from the same wave;
  rebalancing   from the current attempt's per-step telemetry, the ownership
                and order of a wide stage are re-solved at a common wave and
                applied as their own attempt when the predicted gain pays for
                the restart (CONTROL-I21).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import queue
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeGuard

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from wantrain.fault_tolerance.checkpoint import (
    OWNERSHIP_FIELDS,
    CheckpointStore,
    patch_spec_for_resume,
    slot_peer_pins,
)
from wantrain.fault_tolerance.liveness import LivenessJudge, SlotState
from wantrain.fault_tolerance.replacement import find_replacement
from wantrain.model.gpt import PRESETS, GPTConfig, make_config
from wantrain.resources.links import Emulation
from wantrain.resources.node import NodeInfo, capacity_source, class_counts, mixed_fleet
from wantrain.runtime import optim
from wantrain.runtime.exchange import CHUNK_BYTES as EXCHANGE_CHUNK_BYTES
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler import admission, fleetplan, placement
from wantrain.scheduler.linkmemory import LinkMemory
from wantrain.scheduler.partition import best_chain, order_stages
from wantrain.scheduler.selection import SelectionPolicy, node_key, select_nodes
from wantrain.transport import identity
from wantrain.transport.control import (
    Channel,
    ChannelClosed,
    check_mac,
    check_session_mac,
    control_accept,
)
from wantrain.transport.secure import hardware_aes
from wantrain.worker.agent import HANDSHAKE_MSG_MAX, is_loopback

# How far a fully measured hop may fall below its quick probe before the chain
# is re-routed on the better number.
DISAGREE = 0.5
FIT_ATTEMPTS = 4
FIT_HEADROOM = 0.15
# A measured fit overshoot lowers the overflowing node's memory, as the planner sees it, by this multiple of the
# excess: the analytic model misses it by a per-layer amount, so one round usually clears it (shrink_for_overshoot).
MEM_FEEDBACK = 1.25
# A live per-step summary (TELEM_STEP) is accepted only in this shape (CONTROL-I18).
TELEM_MAX_BYTES = 4096
TELEM_WINDOW = 64
TELEM_KEYS = frozenset(
    ("v", "step", "rank", "peer", "step_ms", "emit_us", "ops", "stall_ms", "hops")
)
TELEM_OPS = frozenset(("F", "B", "W", "OPT"))
TELEM_OP_KEYS = frozenset(("n", "med_ms"))
TELEM_HOP_KEYS = frozenset(("frames", "bytes", "busy_ms"))
TELEM_MAX_MS = 1e6
TELEM_MAX_COUNT = 10**12
# A worker controls what some log lines quote: each quoted value is clipped to
# LOG_CLIP characters and the coordinator keeps its last EVENTS_MAX records, so
# no message flood or 16 MiB header can grow its memory (CONTROL-I18).
LOG_CLIP = 120
EVENTS_MAX = 10_000
# The control handshake, CHALLENGE and HELLO of one connection must finish in
# this time; a silent dialer holds one thread and one socket no longer.
HANDSHAKE_TIMEOUT_S = 30.0
# Rebalancing (CONTROL-I21, SPEC-control task 2): re-solved only from a window
# of at least this many summaries per slot, and applied only when the re-solved
# step is at least REPLAN_MIN_GAIN shorter.
REPLAN_MIN_SUMMARIES = 4
REPLAN_MIN_GAIN = 0.10
# Tripwire pieces-unarmed (HANDOUT-M2b, interface I-9): until every leg of
# shard 'pieces' and wire 'native' had landed and F1/F8 passed, run_job refused
# both. arm-pieces armed it after F1 (f1_rotation_a/b), F8 (e3_ x3), e1_native_
# per recipe and native_width1 passed on the integrated tree; the refusal path
# stays tested with the flag patched off.
PIECES_ARMED = True
# Tripwire m4-unarmed (HANDOUT-M4, I-16): until every leg of arch 'hf' and of
# MoE had landed (control-m4, model-core, stage-m4, transport-tied,
# scheduler-m4, agent-hf), run_job, `wantrain local` and `wantrain stage`
# refused both. arm-m4 armed it in wave 2, beside the end-to-end proofs
# (test_m4_coordinator_*) that the post-wave ladder and GATES-M4 run; the
# refusal path stays tested with the flag patched off.
M4_ARMED = True
# The hf record builder child (I-10) and every GATE child (I-9) get this long.
HF_CHILD_TIMEOUT_S = 300.0


def _clip(text: str, n: int = LOG_CLIP) -> str:
    return text if len(text) <= n else text[:n] + f"... ({len(text)} chars)"


class StartRefused(RuntimeError):
    """The coordinator will not listen under this configuration (CONTROL-I16)."""


class _Refused(Exception):
    """A dialer that completed the handshake but is not admitted (REJECT)."""


class WorkerLost(RuntimeError):
    def __init__(self, node_id, why):
        super().__init__(f"worker {node_id}: {why}")
        self.node_id = node_id


@dataclass
class CoordinatorConfig:
    state_dir: str = "wantrain-state"
    token: str | None = None
    heartbeat_timeout_s: float = 15.0
    reconnect_grace_s: float = 30.0
    stall_floor_s: float = 90.0
    stall_steps: int = 6
    strike_gap_s: float = 8.0
    first_step_timeout_s: float = 600.0
    max_heals: int = 2
    max_restarts: int = 2
    max_setup_swaps: int = 2
    poll_s: float = 0.25
    selection: SelectionPolicy = field(default_factory=SelectionPolicy)
    replacement_match: str = "series"
    probe: bool = True
    probe_bulk_bytes: int = 4 << 20
    measure_bulk_bytes: int = 8 << 20
    fit_check: bool = True
    dial_timeout_s: float = 300.0
    request_timeout_s: float = 600.0
    max_run_s: float = 0.0
    dead_host_ttl_s: float = 6 * 3600.0
    default_ckpt_every: int = 1
    exit_report_grace_s: float = 10.0
    # Quiet allowed after every slot finished the final step (final wave,
    # loss gather, drain, last ship) before the run is failed.
    shutdown_limit_s: float = 600.0
    # <run_dir>/telem_steps.jsonl stops growing at this size (one warning).
    telem_log_max_bytes: int = 64 << 20
    # Zero-bubble schedule (scheduler/zero_bubble.py): every slot's F/B/W is
    # profiled on its own node and the schedule is solved for those costs and
    # the measured hops. If profiling fails the analytic schedule stays.
    zb_mem_mode: str = "auto"
    profile_reps: int = 3
    # Rebalancing (CONTROL-I21): at most max_replans per run; the restart cost
    # R is the run's last measured restore-to-first-step, else this value.
    max_replans: int = 2
    replan_restart_cost_s: float = 30.0
    # RUNTIME-I26 / CONTROL-I27: re-cut a width-1 builtin chain from its measured stage costs when the slowest
    # stage shrinks by recut_min_gain and the predicted saving over the remaining steps beats recut_cost_s (a fit
    # of the new cut, the waves converted and restored, the warm-up). A stage whose compute per microbatch stays
    # above slow_ratio x its own baseline for slow_runs summaries is replaced by a spare of the run's class, or,
    # with none, re-cut away from.
    recut: bool = True
    recut_min_gain: float = 0.10
    recut_cost_s: float = 600.0
    slow_ratio: float = 1.3
    slow_runs: int = 3


@dataclass
class Job:
    """Exactly one of `spec` (a recipe to place) or `plan` (a job description
    for the planner) is given. A plan job runs on `devices` workers (default
    `stages`); the planner decides the depth and the width of every stage."""

    spec: JobSpec | None = None
    plan: dict | None = None
    stages: int = 2
    run_id: str | None = None
    devices: int | None = None


@dataclass
class RunResult:
    ok: bool
    run_id: str
    reason: str | None = None
    losses: dict | None = None
    run_dir: str = ""
    attempts: int = 0
    heals: list = field(default_factory=list)
    restarts: int = 0
    rationale: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    replans: list = field(default_factory=list)


class Worker:
    """Coordinator-side record of one registered agent."""

    def __init__(
        self,
        info: NodeInfo,
        ch: Channel,
        session: str,
        emulation: Emulation | None,
        peer_id: str | None = None,
    ):
        self.info = info
        self.ch = ch
        self.session = session
        # The key this node proved in its control handshake: a reconnect is
        # accepted only from it (CONTROL-I16).
        self.peer_id = peer_id
        self.emulation = emulation
        self.connected = True
        self.disconnected_at: float | None = None
        self.last_heartbeat = time.time()
        self.hb: dict = {}
        self.run_id: str | None = None
        self.excluded = False

    @property
    def node_id(self) -> str:
        return self.info.node_id

    def last_seen(self) -> float:
        rx = self.ch.last_rx if self.ch is not None else 0.0
        return max(self.last_heartbeat, rx)


class _Run:
    def __init__(self, run_id: str, run_dir: str):
        self.run_id = run_id
        self.run_dir = run_dir
        self.spec_json = ""
        self.spec: JobSpec | None = None
        self.slots: dict = {}  # (r, p) -> node_id
        self.states: dict = {}  # (r, p) -> SlotState
        self.attempt = 0
        self.heals: list = []
        self.restarts = 0
        self.results: dict = {}
        self.store = CheckpointStore(os.path.join(run_dir, "ckpt"))
        self.lock = threading.RLock()
        self.rationale: dict = {}
        # Live per-step summaries per slot of the current attempt only
        # (CONTROL-I18, BA-23): _launch clears it; the rebalance rule reads it.
        self.telem_window: dict = {}  # (r, p) -> deque of summaries
        # Rebalancing (CONTROL-I21): applied plans, the ownership they set
        # (None: the launch spec's), the newest wave the rule was evaluated at,
        # and the measured restore-to-first-step cost.
        self.replans: list = []
        self.ownership: dict | None = None
        self.replan_checked = -1
        # RUNTIME-I26 / CONTROL-I27: the hops the plan was solved for (a re-cut re-solves with them), each slot's
        # baseline compute per microbatch and the summaries counted since, per attempt.
        self.plan_hops: list = []
        self.slot_base: dict = {}
        self.slot_hist: dict = {}
        self.slow_noted = None  # the slow slot (or re-cut evaluation) already logged this attempt
        self.restore_t0: float | None = None
        self.restart_cost_s: float | None = None
        self.telem_stats = {"accepted": 0, "dropped": 0, "handle_us_max": 0.0}
        self.telem_log_bytes = 0
        self.telem_log_full = False
        # A pieces run's admission inputs, re-read before every rebalance
        # (BA-44): node_id -> plan-fleet node, and whether S2 refuses (plan
        # mode) or warns (spec mode).
        self.pieces_nodes: dict = {}
        self.pieces_s2_refuses = False
        # An arch 'hf' run's admission record (CONTROL-I24), set at the first
        # launch and never rebuilt: the builder child's output and
        # fingerprint, the RECORD text, every node's gate reports and
        # fingerprint, and each slot's recorded device class, torch RECORD
        # and native set. None for every other arch.
        self.hf: dict | None = None


class Coordinator:
    def __init__(
        self,
        # workers dial in over the WAN; token-gated
        host: str = "0.0.0.0",  # nosec B104
        port: int = 0,
        config: CoordinatorConfig | None = None,
    ):
        self.cfg = config or CoordinatorConfig()
        self.host = host
        self.port = port
        # node_id -> activation budget (bytes) a measured fit calibrated for the schedule (CONTROL-I26)
        self._sched_cap: dict = {}
        # RUNTIME-I25: node_ids a measured fit sent back to recomputing
        self._recompute_on: set = set()
        # TRANSPORT-I11: flows per probe and hop measurement, the run's p2p_streams (set per run in run_job)
        # SCHEDULER-I24: node_id -> (entries, measured peak, budget) of its last fit
        self._fit_peaks: dict = {}
        self._probe_streams = 1
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        self.key_path = os.path.join(self.cfg.state_dir, "coord_key")
        self.coord_id: str | None = None
        self.memory = LinkMemory(
            os.path.join(self.cfg.state_dir, "memory"), dead_ttl_s=self.cfg.dead_host_ttl_s
        )
        self.workers: dict[str, Worker] = {}
        self._lock = threading.RLock()
        # rid -> (node_id the request went to, its reply queue) (CONTROL-I20)
        self._pending: dict[str, tuple[str, queue.Queue]] = {}
        self._registered = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._lst: socket.socket | None = None
        self.run: _Run | None = None
        self.events: list = []

    # ================================================================ logging
    def log(self, level: str, msg: str) -> None:
        rec = (time.time(), level, msg)
        self.events.append(rec)
        if len(self.events) > EVENTS_MAX:
            del self.events[: len(self.events) - EVENTS_MAX]
        print(f"[coord {level}] {msg}", flush=True)

    # ================================================================ registry
    def start(self) -> int:
        """Listen for agents. Without a token only a loopback bind is allowed
        (local mode); the coordinator's key is loaded or created first and its
        id logged as COORDINATOR_ID for the workers to pin (CONTROL-I16)."""
        if not self.cfg.token and not is_loopback(self.host):
            raise StartRefused(
                f"refusing to listen on {self.host} without a token: outside a loopback bind "
                "set --token or WANTRAIN_COORD_TOKEN"
            )
        key = identity.load_or_create_key(self.key_path)
        if key is None:
            raise StartRefused("no coordinator key: the control handshake needs `cryptography`")
        pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.coord_id = identity.peer_id(pub)
        self.log("info", f"COORDINATOR_ID {self.coord_id}")
        lst = socket.socket()
        lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lst.bind((self.host, self.port))
        lst.listen(64)
        lst.settimeout(0.5)
        self._lst = lst
        self.port = lst.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()
        if not self.cfg.token:
            self.log(
                "warn",
                f"no coordinator token (local mode): any process on this host that reaches "
                f"port {self.port} can register as a worker",
            )
        self.log("info", f"listening on {self.host}:{self.port}")
        return self.port

    def stop(self, shutdown_workers: bool = True) -> None:
        if shutdown_workers:
            for w in list(self.workers.values()):
                if w.connected and w.ch is not None:
                    try:
                        w.ch.send({"kind": "SHUTDOWN"})
                    except ChannelClosed:
                        pass
        self._stop.set()
        if self._lst is not None:
            self._lst.close()

    def _accept_loop(self) -> None:
        lst = self._lst
        assert lst is not None
        while not self._stop.is_set():
            try:
                sock, addr = lst.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                # A peer that reset right after accept (EINVAL on macOS): drop
                # its socket and keep accepting.
                sock.close()
                continue
            threading.Thread(target=self._handshake, args=(sock, addr), daemon=True).start()

    def _handshake(self, sock, addr) -> None:
        """Control handshake, then CHALLENGE, HELLO and WELCOME, all inside the
        sealed channel (CONTROL-I16): the worker's key comes from the handshake,
        its token MAC and session MAC are bound to the handshake's transcript."""
        ch: Channel | None = None
        try:
            sock.settimeout(HANDSHAKE_TIMEOUT_S)
            ch, info = control_accept(sock, self.key_path, aes_hw=hardware_aes())
            ch.send({"kind": "CHALLENGE", "auth": bool(self.cfg.token)})
            # No file_dir: a file body before admission is refused unread.
            hello, _ = ch.recv(max_inline=HANDSHAKE_MSG_MAX)
            sock.settimeout(None)
            if hello.get("kind") != "HELLO":
                raise ChannelClosed("first message was not HELLO")
            transcript = info["transcript"]
            if self.cfg.token and not check_mac(self.cfg.token, transcript, hello.get("mac")):
                raise _Refused("bad token")
            hello.setdefault("advertise_host", addr[0])
            w = self._register(hello, ch, info["peer_id"], transcript)
        except _Refused as e:
            assert ch is not None
            try:
                ch.send({"kind": "REJECT", "reason": str(e)})
            except ChannelClosed:
                pass
            ch.close()
            self.log("warn", f"rejected a worker from {addr[0]}: {e}")
            return
        except (ChannelClosed, OSError, ValueError, KeyError) as e:
            self.log("warn", f"handshake from {addr[0]} failed: {_clip(str(e))}")
            if ch is not None:
                ch.close()
            else:
                sock.close()
            return
        self._conn_loop(w, ch)

    def _register(self, hello: dict, ch: Channel, peer: str, transcript: bytes) -> Worker:
        with self._lock:
            nid = hello.get("node_id")
            w = self.workers.get(nid) if isinstance(nid, str) else None
            if w is not None:
                # A reconnect proves the session over this transcript, never
                # sends it, and only from the key the node registered with.
                if w.peer_id != peer:
                    raise _Refused(f"reconnect of {nid} from key {peer}, not its registered key")
                if not check_session_mac(w.session, transcript, hello.get("session_mac")):
                    raise _Refused(f"reconnect of {nid} without a valid session MAC")
                old = w.ch
                w.ch, w.connected, w.disconnected_at = ch, True, None
                w.last_heartbeat = time.time()
                if old is not None and old is not ch:
                    old.close()
                ch.send({"kind": "WELCOME", "node_id": nid, "session": w.session})
                self.log("info", f"worker {w.info.name} ({nid}) reconnected")
                self._registered.notify_all()
                return w
            nid = uuid.uuid4().hex[:10]
            info = NodeInfo.from_hello(nid, hello)
            emu = None
            if hello.get("emulation"):
                e = hello["emulation"]
                emu = Emulation(e.get("region"), e.get("profile"))
                info.region = info.region or e.get("region")
            w = Worker(info, ch, secrets.token_hex(16), emu, peer_id=peer)
            self.workers[nid] = w
            ch.send({"kind": "WELCOME", "node_id": nid, "session": w.session})
            decl = f" declared={info.declared}" if info.declared else ""
            self.log(
                "info",
                f"registered {info.name} ({nid}) key={peer} {info.host}:{info.data_port} "
                f"{info.device}/{info.gpu_name} {info.speed():.3f} TFLOPs "
                f"{info.memory_bytes() / 2**30:.1f} GB region={info.region}{decl}",
            )
            self._registered.notify_all()
            return w

    def _conn_loop(self, w: Worker, ch: Channel) -> None:
        inbox = os.path.join(self.cfg.state_dir, "incoming")
        try:
            while not self._stop.is_set():
                msg, body = ch.recv(file_dir=inbox)
                try:
                    self._dispatch(w, msg, body)
                except Exception as e:  # noqa: BLE001 -- one bad message never ends the reader
                    # CONTROL-I18: a reader that died here would leave a live
                    # worker's heartbeats unread until it is convicted.
                    self.log(
                        "warn",
                        f"malformed {_clip(str(msg.get('kind')))} from {w.info.name}: "
                        f"{_clip(repr(e))}",
                    )
                finally:
                    # CKPT_PART and TELEMETRY moved the bodies they keep; any other
                    # body (on any kind, a request's reply included, which
                    # _request discards) is unlinked here, so none fills the disk.
                    if isinstance(body, str) and os.path.exists(body):
                        os.unlink(body)
        except (ChannelClosed, OSError, ValueError) as e:
            with self._lock:
                if w.ch is ch:
                    w.connected = False
                    w.disconnected_at = time.time()
                    self.log("warn", f"worker {w.info.name} ({w.node_id}) disconnected: {e}")
                    self._sync_slot_connection(w)
                    self._drop_staged_parts(w)
        finally:
            # A stopped coordinator's reader ends here too: its channel closes
            # with it, never left to the GC.
            ch.close()

    def _sync_slot_connection(self, w: Worker) -> None:
        run = self.run
        if run is None:
            return
        with run.lock:
            for slot, nid in run.slots.items():
                if nid == w.node_id and slot in run.states:
                    st = run.states[slot]
                    st.connected = w.connected
                    st.disconnected_at = w.disconnected_at

    def _drop_staged_parts(self, w: Worker) -> None:
        """Staged CKPT_PARTs belong to the connection that sent them: a wave
        cut by a disconnect is shipped again whole on the next one."""
        run = self.run
        if run is None:
            return
        for slot, nid in list(run.slots.items()):
            if nid == w.node_id:
                run.store.drop_staged(slot)

    # ================================================================ messages
    def _dispatch(self, w: Worker, msg: dict, body) -> None:
        kind = msg["kind"]
        rid = msg.get("rid")
        if kind == "HEARTBEAT":
            w.last_heartbeat = time.time()
            w.hb = msg
            self._slot_update(w, msg, heartbeat=True)
            return
        pending = self._pending.get(rid) if rid else None
        if pending is not None:
            # Only the asked worker's connection answers its request (CONTROL-I20);
            # a drop raises, so _conn_loop logs it and unlinks the body.
            node_id, q = pending
            if node_id != w.node_id:
                raise ValueError(f"reply {_clip(str(rid))} answers a request to another worker")
            q.put((msg, body))
            return
        run = self.run
        if run is None or msg.get("run_id") != run.run_id:
            return  # _conn_loop unlinks any body
        if kind == "PROGRESS":
            self._slot_update(w, msg, progress=True)
        elif kind == "CKPT_PART":
            # One file of a wave, staged for the connection's slot and attempt
            # (CONTROL-I20, BA-43); a drop raises, so _conn_loop logs it and
            # unlinks the body.
            slot, step = self._wave_slot(w, run, msg)
            if not msg.get("file") or not isinstance(body, str) or not os.path.isfile(body):
                raise ValueError("no file body")
            run.store.stage_part(slot, run.attempt, step, msg.get("name"), msg.get("sha256"), body)
        elif kind == "CKPT":
            # The manifest that commits the staged parts as one wave, after
            # every hash and its step verify (CONTROL-I23); otherwise the staged
            # parts are dropped and the CKPT is logged.
            slot, step = self._wave_slot(w, run, msg)
            if run.store.commit_staged(slot, run.attempt, step, msg.get("manifest")):
                run.store.prune(list(run.slots))
        elif kind == "TELEMETRY":
            # Rank, peer and attempt come from the connection's slot, never from
            # the message (CONTROL-I18); a drop raises, so _conn_loop logs it
            # and unlinks the body.
            _, slot = self._slot_for(w, msg)
            if slot is None:
                raise ValueError("no slot of the current attempt on this connection")
            if not msg.get("file") or not isinstance(body, str) or not os.path.isfile(body):
                raise ValueError("no file body")
            r, p = slot
            os.replace(body, os.path.join(run.run_dir, f"telemetry_r{r}p{p}_a{run.attempt}.jsonl"))
        elif kind == "TELEM_STEP":
            self._telem_step(w, run, msg)
        elif kind == "EXIT":
            self._slot_update(w, msg, exit=True)
        elif kind == "RESULT":
            run.results[self._own_slot(w, run, msg)] = msg

    def _telem_step(self, w: Worker, run: _Run, msg: dict) -> None:
        """Intake of one live per-step summary (CONTROL-I18): validated against
        the connection's slot, kept in its window and logged to
        <run_dir>/telem_steps.jsonl. Only the _monitor thread's rebalance rule
        reads the window (BA-45); nothing here decides anything."""
        t0 = time.perf_counter()
        summary = msg.get("summary")
        why: str | None
        with run.lock:
            try:
                _, slot = self._slot_for(w, msg)
                st = run.states.get(slot) if slot is not None else None
                if st is None:
                    why = "no slot of the current attempt on this connection"
                else:
                    why = _telem_summary_error(summary, slot, st.start_step, self._final_step(run))
            except Exception as e:  # noqa: BLE001 -- any malformed summary is a counted drop
                why = f"malformed ({_clip(repr(e))})"
            stats = run.telem_stats
            # isinstance narrows summary for the type checker; the validator
            # already refuses a non-dict, so the fallback text is defensive only.
            if why is None and isinstance(summary, dict):
                stats["accepted"] += 1
                run.telem_window.setdefault(slot, deque(maxlen=TELEM_WINDOW)).append(summary)
                self._log_telem(run, dict(summary, attempt=run.attempt))
            else:
                stats["dropped"] += 1
                self.log(
                    "warn",
                    f"dropped TELEM_STEP from {w.info.name}: {why or 'summary is not an object'}",
                )
            stats["handle_us_max"] = max(stats["handle_us_max"], (time.perf_counter() - t0) * 1e6)
            run.rationale["telemetry"] = stats

    def _log_telem(self, run: _Run, rec: dict) -> None:
        if run.telem_log_full:
            return
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        if run.telem_log_bytes + len(line) > self.cfg.telem_log_max_bytes:
            run.telem_log_full = True
            self.log(
                "warn",
                f"telem_steps.jsonl reached telem_log_max_bytes "
                f"({self.cfg.telem_log_max_bytes}); no more summaries are written",
            )
            return
        with open(os.path.join(run.run_dir, "telem_steps.jsonl"), "a") as f:
            f.write(line)
        run.telem_log_bytes += len(line)

    def _slot_for(self, w: Worker, msg: dict):
        run = self.run
        if run is None or msg.get("run_id") != run.run_id:
            return None, None
        if int(msg.get("attempt", -1)) != run.attempt:
            return None, None
        for slot, nid in run.slots.items():
            if nid == w.node_id:
                return run, slot
        return None, None

    def _wave_slot(self, w: Worker, run: _Run, msg: dict) -> tuple:
        """(slot, step) of a CKPT_PART or CKPT: the slot is the connection's,
        of the current attempt (CONTROL-I20); a rank or peer the message names
        anyway must be that slot's. Raises ValueError otherwise."""
        slot = self._conn_slot(w, run, msg)
        for k, v in (("rank", slot[0]), ("peer", slot[1])):
            if k in msg and (not _is_int(msg[k]) or msg[k] != v):
                raise ValueError(
                    f"names {k} {_clip(repr(msg[k]))}; the connection holds slot {slot}"
                )
        step = msg.get("step")
        if not _is_int(step) or step < 0:
            raise ValueError(f"step {_clip(repr(step))} is not a non-negative int")
        return slot, step

    def _conn_slot(self, w: Worker, run: _Run, msg: dict) -> tuple:
        """The slot of the current attempt bound to this connection; the
        message's attempt must be exactly that attempt (CONTROL-I20)."""
        attempt = msg.get("attempt")
        if not _is_int(attempt) or attempt != run.attempt:
            raise ValueError(f"attempt {_clip(repr(attempt))} is not attempt {run.attempt}")
        _, slot = self._slot_for(w, msg)
        if slot is None:
            raise ValueError("no slot of the current attempt on this connection")
        return slot

    def _own_slot(self, w: Worker, run: _Run, msg: dict) -> tuple:
        """The slot of the current attempt bound to this connection, for a
        message that also names its rank, peer and attempt (CONTROL-I20).
        Raises ValueError unless the message names exactly that slot and
        attempt."""
        slot = self._conn_slot(w, run, msg)
        rank, peer = msg.get("rank"), msg.get("peer")
        if not (_is_int(rank) and _is_int(peer)) or (rank, peer) != slot:
            raise ValueError(
                f"names slot ({_clip(repr(rank))}, {_clip(repr(peer))}); "
                f"the connection holds slot {slot}"
            )
        return slot

    def _slot_update(self, w, msg, heartbeat=False, progress=False, exit=False) -> None:
        run, slot = self._slot_for(w, msg)
        if run is None:
            return
        with run.lock:
            st = run.states.get(slot)
            if st is None:
                return
            if heartbeat:
                st.last_heartbeat = time.time()
                st.stage_alive = bool(msg.get("stage_alive"))
            # Only WT_STEP is an optimizer step this attempt ran; WT_METRIC
            # lines are replayed in a burst on every resume (CONTROL-I17).
            if progress and msg.get("src") == "step":
                step = msg.get("step")
                if isinstance(step, int) and not isinstance(step, bool):
                    st.add_progress(step, time.time(), self._final_step(run))
                else:
                    self.log("warn", f"slot {slot}: dropped progress with step {_clip(repr(step))}")
            if exit:
                st.exit_code = int(msg["rc"])
                st.done_marker = bool(msg.get("done_marker"))
                st.log_tail = msg.get("log_tail", "")
                st.stage_alive = False
                lvl = "info" if st.exit_code == 0 else "warn"
                self.log(
                    lvl,
                    f"slot {slot} exited {st.exit_code}"
                    + (
                        ""
                        if st.exit_code == 0
                        else f": {_clip(str(st.log_tail.strip().splitlines()[-1:] or ''))}"
                    ),
                )

    def _request(
        self,
        w: Worker,
        msg: dict,
        body: bytes = b"",
        timeout: float | None = None,
        file: str | None = None,
        reply_kind: str | None = None,
    ) -> dict:
        rid = uuid.uuid4().hex
        q: queue.Queue = queue.Queue()
        self._pending[rid] = (w.node_id, q)
        try:
            if not w.connected or w.ch is None:
                raise WorkerLost(w.node_id, "not connected")
            m = dict(msg, rid=rid)
            try:
                if file:
                    w.ch.send_file(m, file)
                else:
                    w.ch.send(m, body)
            except ChannelClosed as e:
                raise WorkerLost(w.node_id, f"send failed ({e})") from e
            deadline = time.time() + (timeout or self.cfg.request_timeout_s)
            while True:
                try:
                    reply, _ = q.get(timeout=0.5)
                    break
                except queue.Empty:
                    if (
                        not w.connected
                        and time.time() - (w.disconnected_at or 0) > self.cfg.reconnect_grace_s
                    ):
                        raise WorkerLost(w.node_id, "disconnected while answering") from None
                    if time.time() > deadline:
                        raise WorkerLost(w.node_id, f"no reply to {msg['kind']}") from None
            if reply.get("kind") == "ERROR":
                raise RuntimeError(f"{w.info.name}: {msg['kind']} failed: {reply.get('error')}")
            return reply
        finally:
            self._pending.pop(rid, None)

    # ================================================================ discovery
    def idle_workers(self) -> list[Worker]:
        with self._lock:
            return [
                w
                for w in self.workers.values()
                if w.connected and not w.excluded and w.run_id is None
            ]

    def wait_for_workers(self, n: int, timeout: float = 300.0) -> list[Worker]:
        deadline = time.time() + timeout
        with self._registered:
            while len(self.idle_workers()) < n:
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError(
                        f"only {len(self.idle_workers())} of {n} workers "
                        f"registered within {timeout:.0f}s"
                    )
                self._registered.wait(min(left, 1.0))
        return self.idle_workers()

    # ================================================================ routing
    def _probe_matrix(self, ws: list[Worker]):
        targets = [[w.info.host, w.info.probe_port, w.info.region] for w in ws]
        rows: list = [None] * len(ws)
        errs: list = []

        def one(i, w):
            try:
                r = self._request(
                    w,
                    {
                        "kind": "PROBE",
                        "targets": targets,
                        "self_index": i,
                        "bulk_bytes": self.cfg.probe_bulk_bytes,
                        "streams": self._probe_streams,
                    },
                )
                rows[i] = r["row"]
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        # Sequential rows: concurrent bulk probes would measure each other.
        for i, w in enumerate(ws):
            one(i, w)
        if errs:
            raise errs[0]
        n = len(ws)
        rtt = [[math.inf] * n for _ in range(n)]
        bw = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    rtt[i][j], bw[i][j] = 0.0, math.inf
                    continue
                rv, bv = rows[i]["rtt_ms"][j], rows[i]["bw_mbps"][j]
                rtt[i][j] = math.inf if rv is None else float(rv)
                bw[i][j] = 0.0 if bv is None else float(bv)
        # Symmetrise on the slower direction: a hop carries both ways.
        for i in range(n):
            for j in range(i + 1, n):
                b = min(bw[i][j], bw[j][i])
                r = max(rtt[i][j], rtt[j][i])
                bw[i][j] = bw[j][i] = b
                rtt[i][j] = rtt[j][i] = r
        return rtt, bw

    def _route_chain(self, ws: list[Worker], run_id: str):
        """Probe the mesh, order the chain by its worst hop, then fully
        measure each chosen hop and re-route while a hop measures far below
        its probe (port of the source's measure-and-reroute loop)."""
        n = len(ws)
        rtt, bw = self._probe_matrix(ws)
        self.log(
            "info",
            "probe matrix (Mbps): "
            + json.dumps([[None if math.isinf(x) else round(x, 1) for x in row] for row in bw]),
        )
        order = order_stages(rtt, bw)
        measured: dict = {}
        # Which end dialed each measured pair: a re-routed chain may cross the
        # pair the other way, and its per-direction figures must follow it.
        dialer: dict = {}
        for attempt in range(1, 4):
            misled = False
            for a, b in zip(order, order[1:]):
                key = tuple(sorted((a, b)))
                if key in measured:
                    continue
                client, server = ws[a], ws[b]
                rep = self._request(
                    client,
                    {
                        "kind": "MEASURE",
                        "host": server.info.host,
                        "port": server.info.probe_port,
                        "region": server.info.region,
                        "bulk_bytes": self.cfg.measure_bulk_bytes,
                        "streams": self._probe_streams,
                    },
                )
                if not rep.get("ok"):
                    self.log(
                        "warn",
                        f"hop {client.info.name}->{server.info.name} carries no "
                        f"traffic ({rep.get('error')}); routing around it",
                    )
                    bw[a][b] = bw[b][a] = 0.0
                    misled = True
                    break
                m = rep["link"]
                measured[key] = m
                dialer[key] = a
                self.log(
                    "info",
                    f"link {client.info.name}->{server.info.name}: fwd "
                    f"{m['forward_mbps']} rev {m['reverse_mbps']} bidir "
                    f"{m['bidir_forward_mbps']}+{m['bidir_reverse_mbps']} Mbps, "
                    f"RTT {m['rtt_ms']} ms, duplex {m['duplex']}",
                )
                probed = bw[a][b]
                bw[a][b] = bw[b][a] = float(m["mbps"])
                rtt[a][b] = rtt[b][a] = float(m["rtt_ms"])
                if probed > 0 and m["mbps"] < probed * DISAGREE:
                    misled = True
            if not misled:
                break
            nxt = order_stages(rtt, bw)
            if nxt == order:
                break
            self.log(
                "info",
                f"re-routing (attempt {attempt + 1}): a hop measured far below "
                f"its probe; new order {[ws[i].info.name for i in nxt]}",
            )
            order = nxt
        hops = []
        for a, b in zip(order, order[1:]):
            key = tuple(sorted((a, b)))
            m = measured.get(key)
            hops.append(m if m is None or dialer[key] == a else _reversed_hop(m))
        if any(h is None for h in hops):
            _o, bottleneck = best_chain(
                n, [(i, j, bw[i][j]) for i in range(n) for j in range(i + 1, n)]
            )
            raise RuntimeError(
                f"no chain order connects all {n} nodes with measured hops "
                f"(best bottleneck {bottleneck:.1f} Mbps)"
            )
        done = [h for h in hops if h is not None]
        for (a, b), h in zip(zip(order, order[1:]), done):
            if ws[a].info.host != ws[b].info.host:
                self.memory.record_path(
                    ws[a].info.host, ws[b].info.host, h["mbps"], h["rtt_ms"], run_id
                )
        worst = min(done, key=lambda h: h["mbps"]) if done else None
        return order, done, worst

    # ================================================================ jobs
    def run_job(self, job: Job, wait_workers_s: float = 300.0) -> RunResult:
        run_id = job.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(self.cfg.state_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        run = _Run(run_id, run_dir)
        self.run = run
        self.log("info", f"run {run_id}: starting")
        unarmed = unarmed_reason(job)
        if unarmed is not None:
            self.log("error", f"run {run_id} refused: {unarmed}")
            return self._finish(run, False, unarmed)
        try:
            for swap in range(self.cfg.max_setup_swaps + 1):
                try:
                    groups = self._setup(run, job, wait_workers_s)
                    break
                except WorkerLost as e:
                    self._exclude(e.node_id, f"lost during setup: {e}", run_id)
                    if swap == self.cfg.max_setup_swaps:
                        raise
                    self.log("warn", f"{e}; re-selecting the fleet (setup swap {swap + 1})")
            self._launch(run, groups)
            return self._monitor(run)
        except Exception as e:  # noqa: BLE001 -- the run fails, the coordinator survives
            self.log("error", f"run {run_id} failed: {type(e).__name__}: {e}")
            self._abort_all(run)
            return self._finish(run, False, f"{type(e).__name__}: {e}")

    def _exclude(self, node_id: str, why: str, run_id: str) -> None:
        w = self.workers.get(node_id)
        if w is None:
            return
        w.excluded = True
        w.run_id = None
        self.memory.record_dead(node_key(w.info), why, run_id, w.info.host)

    def _setup(self, run: _Run, job: Job, wait_workers_s: float):
        if job.spec is not None:
            template = JobSpec.from_json(job.spec.to_json())
            widths = template.stage_widths or [1] * template.stages
            need = sum(widths)
        else:
            if not job.plan:
                raise ValueError("a job needs a spec or a plan")
            # The planner decides the widths; the workers are routed as a chain.
            template, need = None, job.devices or job.stages
            widths = [1] * need
        self.wait_for_workers(need, wait_workers_s)
        pool = [w.info for w in self.idle_workers()]
        sel = select_nodes(pool, need, self.cfg.selection, self.memory)
        if not sel.ok:
            raise RuntimeError(f"no fleet: {sel.reason}")
        # CONTROL-I25 at the first launch's admission, before anything is placed.
        mixed = mixed_fleet((n.name, n.device_class) for n in sel.chosen)
        if mixed is not None:
            raise RuntimeError(f"admission refused (CONTROL-I25): {mixed}")
        self.log("info", f"selection: {sel.note}")
        chosen = [self.workers[n.node_id] for n in sel.chosen]
        for w in chosen:
            w.run_id = run.run_id

        chain_only = all(x == 1 for x in widths)
        hops, worst = [], None
        # TRANSPORT-I11: probe and measure each hop with the data plane's own flow count
        plan_streams = (job.plan or {}).get("p2p_streams") if job.plan else None
        self._probe_streams = max(
            1, int(plan_streams or (job.spec.p2p_streams if job.spec else 1) or 1)
        )
        if chain_only and need > 1 and self.cfg.probe and all(w.info.probe_port for w in chosen):
            order, hops, worst = self._route_chain(chosen, run.run_id)
            chosen = [chosen[i] for i in order]
        elif chain_only and template is not None:
            mo = placement.memory_order(template, [w.info for w in chosen])
            if mo is not None:
                chosen = [chosen[i] for i in mo]
        emus = {w.node_id: w.emulation for w in chosen}

        if template is not None:
            groups, k = [], 0
            for wd in widths:
                groups.append(chosen[k : k + wd])
                k += wd
            info_groups = [[w.info for w in g] for g in groups]
            if getattr(template.model, "arch", None) == "hf":
                # CONTROL-I24, BA-424/425: record and GATE before placement.
                m = template.model
                out = self._hf_first_launch(
                    run,
                    getattr(m, "hf_config", None),
                    {"compute_dtype": m.compute_dtype, "seq_len": m.seq_len},
                    template.init_seed,
                    template.mb_size,
                    chosen + self._class_spares(chosen[0].info.device_class),
                )
                _hf_pin(template.model, out)
                # I-7's JobSpec.device_classes (stage-m4), set by name.
                setattr(template, "device_classes", self._hf_admit_map(run, info_groups))  # noqa: B010
            spec = template
            spec.tag = spec.tag if spec.tag not in (None, "", "run") else run.run_id
            if not spec.ckpt_every:
                spec.ckpt_every = self.cfg.default_ckpt_every
            spec.dial_timeout_s = spec.dial_timeout_s or self.cfg.dial_timeout_s
            spec.resume = False
            rationale = placement.place(spec, info_groups, emus if all(emus.values()) else None)
            if spec.stages > 1:
                rationale["schedule"] = self._zero_bubble(
                    spec, info_groups, hops, template.partition is None
                )
            # BA-50: a spec-mode pieces job gets I16's class check and memory
            # terms before ASSIGN; S1 and S2 are warnings.
            run.pieces_nodes = _nodes_by_id([n for g in info_groups for n in g], None)
            run.pieces_s2_refuses = False
            rationale["pieces_warnings"] = self._check_pieces(spec, info_groups, run)
        else:
            infos = [w.info for w in chosen]
            nic = nic_from_hops(infos, hops) if hops else None
            spec, rationale, info_groups = self._plan(run, job, infos, hops, worst, nic, emus)
            groups = [[self.workers[n.node_id] for n in g] for g in info_groups]
            used = {n.node_id for g in info_groups for n in g}
            for w in chosen:
                if w.node_id not in used:
                    w.run_id = None
        rationale["hops"] = hops
        rationale["selection"] = sel.note
        if spec.stages > 1:
            rationale["partition"] = spec.partition
            rationale["layers_per_stage"] = [b - a for a, b in spec.partition]
        run.rationale = rationale
        run.slots = {(r, p): g[p].node_id for r, g in enumerate(groups) for p in range(len(g))}
        pins = slot_peer_pins(self._slot_peers(run))
        spec.slot_peers, spec.trusted_peers = pins["slot_peers"], pins["trusted_peers"]
        run.spec = spec
        run.spec_json = spec.to_json()
        with open(os.path.join(run.run_dir, "spec.json"), "w") as f:
            f.write(run.spec_json)
        with open(os.path.join(run.run_dir, "rationale.json"), "w") as f:
            json.dump(rationale, f, indent=2, default=str)
        self.log("info", f"placement: {rationale['placement']} partition {spec.partition}")
        if spec.ckpt_every <= 0:
            self.log(
                "warn", "ckpt_every=0: a dead worker cannot be healed, only restarted from step 0"
            )
        return groups

    def _check_pieces(self, spec: JobSpec, info_groups, run: _Run) -> list[str]:
        """I16 over a placed pieces spec as it will run (pieces_admission,
        with the run's inputs): raises RuntimeError naming every refusal
        (the class, the peer or the host); returns the warnings' texts."""
        refusals, warnings = pieces_admission(
            spec, info_groups, run.pieces_nodes, run.pieces_s2_refuses
        )
        if refusals:
            raise RuntimeError(
                "shard 'pieces' refused by SCHEDULER-I16: " + "; ".join(r.text() for r in refusals)
            )
        texts = [r.text() for r in warnings]
        for t in texts:
            self.log("warn", f"pieces: {t}")
        return texts

    # ================================================================ arch 'hf'
    def _hf_first_launch(
        self, run: _Run, hf_config, model_recipe: dict, seed: int, mb_size: int, pool: list
    ) -> dict:
        """The first launch of an arch 'hf' run (CONTROL-I24, BA-424/425): the
        record, census, hf_code and RECORD text built once in a child, then
        GATE to every node of the pool (selected nodes and spares) and their
        fingerprints admitted, the builder's included. -> the builder's
        output (I-10). Raises naming every refusal; nothing is placed."""
        from wantrain.model import hfm  # type: ignore[attr-defined]  # model-core's (I-2)

        if not isinstance(hf_config, dict):
            raise RuntimeError("arch 'hf' refused: the model has no hf_config dict")
        recipe = {**model_recipe, "mb_size": mb_size}
        # A setup swap (a node lost while gating) re-gates; it never rebuilds.
        if run.hf is None or run.hf["hf_config"] != hf_config:
            out = self._hf_build_record(run, hf_config, seed)
            text = record_text_of({"transformers": {"record_gz_b64": out["record_text_gz_b64"]}})
            if text is None:
                raise RuntimeError("hf record builder: its RECORD text does not decode")
            run.hf = {
                "hf_config": hf_config,
                "built": out,
                "builder": out["fingerprint"],
                "hf_code": out["hf_code"],
                "record_text": text,
                "reports": {},
                "fingerprints": {},
                "slots": {},
                "fresh": set(),
            }
        hf = run.hf
        hf["recipe"] = recipe
        hf["pool"] = [w.node_id for w in pool]
        self._hf_gate(run, pool)
        refusals = hf_admission(
            hf["reports"],
            hf["fingerprints"],
            hf["builder"],
            {},
            hf["pool"],
            torch_records=hfm.TORCH_RECORDS,
            record_diff=hfm.record_diff,
        )
        self._hf_save(run)
        if refusals:
            raise RuntimeError(
                "arch 'hf' refused before placement (CONTROL-I24): " + "; ".join(refusals)
            )
        self.log("info", f"hf: {len(pool)} node(s) gated and admitted with the record builder")
        return hf["built"]

    def _hf_build_record(self, run: _Run, hf_config: dict, seed: int) -> dict:
        """I-10: `python -X pycache_prefix=<fresh> -m wantrain.model.hfm
        record` in hfm.stage_env, on this machine's CPU; the prefix is
        deleted at exit (BA-442). -> the child's output dict."""
        from wantrain.model import hfm  # type: ignore[attr-defined]  # model-core's (I-2)
        from wantrain.runtime.determinism import pinned_env

        with tempfile.TemporaryDirectory(prefix="wantrain-hf-record-") as td:
            prefix = os.path.join(td, "pycache")
            os.mkdir(prefix)
            inp, outp = os.path.join(td, "in.json"), os.path.join(td, "out.json")
            with open(inp, "w") as f:
                json.dump({"hf_config": hf_config, "seed": seed}, f)
            cmd = [
                sys.executable,
                *hfm.interpreter_args(prefix),
                "-m",
                "wantrain.model.hfm",
                "record",
                "--in",
                inp,
                "--out",
                outp,
            ]
            try:
                p = subprocess.run(
                    cmd,
                    env=hfm.stage_env(pinned_env(dict(os.environ))),
                    capture_output=True,
                    text=True,
                    timeout=HF_CHILD_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"hf record builder: no result within {HF_CHILD_TIMEOUT_S:.0f} s"
                ) from e
            if p.returncode != 0:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
                raise RuntimeError(f"hf record builder failed (exit {p.returncode}): {tail[0]}")
            with open(outp) as f:
                out = json.load(f)
        missing = [
            k
            for k in ("record", "census", "fields", "hf_code", "record_text_gz_b64", "fingerprint")
            if not isinstance(out, dict) or k not in out
        ]
        if missing:
            raise RuntimeError(f"hf record builder: output lacks {missing}")
        self.log("info", f"hf: record built once for run {run.run_id}")
        return out

    def _hf_gate(self, run: _Run, ws: list, tolerate: bool = False) -> set:
        """GATE (I-9) to each worker in parallel; their reports and
        fingerprints replace the run's. -> the node ids gated. A failed
        request raises (the first), or with tolerate drops that node's
        reports (a spare at a heal) and is logged."""
        hf = run.hf
        assert hf is not None
        req = {
            "kind": "GATE",
            "run_id": run.run_id,
            "hf_config": hf["hf_config"],
            "recipe": hf["recipe"],
        }
        got: set = set()
        errs: list = []

        def one(w: Worker) -> None:
            devices = ["cpu"] + ([w.info.device] if w.info.device != "cpu" else [])
            try:
                rep = self._request(
                    w,
                    dict(req, devices=devices),
                    timeout=HF_CHILD_TIMEOUT_S * len(devices) + 60.0,
                )
                reports, fp = rep.get("reports"), rep.get("fingerprint")
                if (
                    rep.get("kind") != "GATE_REPORT"
                    or not isinstance(reports, list)
                    or not isinstance(fp, dict)
                ):
                    raise RuntimeError(f"{w.info.name}: malformed GATE_REPORT")
            except Exception as e:  # noqa: BLE001 -- raised below or logged
                with run.lock:
                    hf["reports"].pop(w.node_id, None)
                    hf["fingerprints"].pop(w.node_id, None)
                errs.append((w, e))
                return
            with run.lock:
                hf["reports"][w.node_id] = reports
                hf["fingerprints"][w.node_id] = fp
                got.add(w.node_id)

        ts = [threading.Thread(target=one, args=(w,)) for w in ws]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs and not tolerate:
            raise errs[0][1]
        for w, e in errs:
            self.log("warn", f"hf: GATE on {w.info.name} failed ({_clip(str(e))}); not a candidate")
        return got

    def _class_spares(self, device_class: str, gpu_name: str | None = None) -> list:
        """The idle workers of `device_class` (CONTROL-I25): the only spares a
        run of that class gates or heals with; none for an empty class. With an
        architecture class (WANTRAIN_DEVICE_CLASS=arch, "cuda:sm_86") and the
        dead slot's `gpu_name`, only that GPU model: the class groups SKUs at
        admission, but a heal onto another SKU would change the device map the
        run's bits belong to (ADR-0005)."""
        same_model = gpu_name is not None and device_class.startswith("cuda:sm_")
        return [
            w
            for w in self.idle_workers()
            if device_class
            and w.info.device_class == device_class
            and (not same_model or w.info.gpu_name == gpu_name)
        ]

    def _hf_admit_map(self, run: _Run, groups) -> list:
        """Decision 8 on the map placement built (groups[stage][peer] =
        NodeInfo): each slot's device class is its node's gate report for
        the slot's device; the map is admitted by hf_admission and recorded
        with each slot's torch RECORD and native set. -> device_classes
        [stage][peer]. Raises naming every refusal."""
        from wantrain.model import hfm  # type: ignore[attr-defined]  # model-core's (I-2)

        hf = run.hf
        assert hf is not None
        slot_map: dict = {}
        for r, g in enumerate(groups):
            for p, n in enumerate(g):
                reps = hf["reports"].get(n.node_id) or []
                cls = next(
                    (
                        x.get("device_class")
                        for x in reps
                        if isinstance(x, dict) and x.get("device") == n.device
                    ),
                    None,
                )
                slot_map[(r, p)] = {"node": n.node_id, "device_class": cls}
        refusals = class_refusals(slot_map, {n.node_id: n for g in groups for n in g})
        refusals += hf_admission(
            hf["reports"],
            hf["fingerprints"],
            hf["builder"],
            slot_map,
            hf["pool"],
            torch_records=hfm.TORCH_RECORDS,
            record_diff=hfm.record_diff,
        )
        if refusals:
            raise RuntimeError(
                "arch 'hf' refused before placement (CONTROL-I24): " + "; ".join(refusals)
            )
        for e in slot_map.values():
            fp = hf["fingerprints"][e["node"]]
            e["torch_records"] = fp["torch"]["records"]
            e["native"] = fp.get("native")
        hf["slots"] = slot_map
        self._hf_save(run)
        return [
            [slot_map[(r, p)]["device_class"] for p in range(len(g))] for r, g in enumerate(groups)
        ]

    def _hf_readmit(self, run: _Run) -> None:
        """CONTROL-I24 at a resume, heal, restart or rebalance: every node of
        the map it would build is gated again (unless gated for this
        admission already), and the map is admitted against the run's
        recorded hf_code, RECORD text and slots; the record is never
        rebuilt. Raises naming every refusal; nothing relaunches."""
        from wantrain.model import hfm  # type: ignore[attr-defined]  # model-core's (I-2)

        hf = run.hf
        assert hf is not None
        fresh, hf["fresh"] = hf["fresh"], set()
        nodes = sorted(set(run.slots.values()))
        self._hf_gate(run, [self.workers[n] for n in nodes if n not in fresh])
        # The reference is the RECORD text kept in the run record (written once
        # at the first launch), read at every admission; a missing one refuses.
        try:
            with open(os.path.join(run.run_dir, "hf_record.txt")) as f:
                kept = f.read()
        except OSError as e:
            raise RuntimeError(
                f"admission refused (CONTROL-I24); nothing relaunched: no kept RECORD text ({e})"
            ) from e
        kw: dict = {
            "torch_records": hfm.TORCH_RECORDS,
            "record_diff": hfm.record_diff,
            "recorded": {"hf_code": hf["hf_code"], "record_text": kept},
        }
        # An idle spare counts toward decision 8's pool only when it is itself
        # admitted; one that is not never blocks the admission of the map.
        spares = [
            w.node_id
            for w in self.idle_workers()
            if w.node_id in hf["reports"]
            and not hf_admission(
                hf["reports"], hf["fingerprints"], hf["builder"], {}, [w.node_id], **kw
            )
        ]
        slot_map = {slot: dict(hf["slots"][slot], node=nid) for slot, nid in run.slots.items()}
        refusals = class_refusals(slot_map, {n: self.workers[n].info for n in nodes})
        refusals += hf_admission(
            hf["reports"],
            hf["fingerprints"],
            hf["builder"],
            slot_map,
            nodes + spares,
            **kw,
        )
        if refusals:
            raise RuntimeError(
                "admission refused (CONTROL-I24); nothing relaunched: " + "; ".join(refusals)
            )
        hf["slots"] = slot_map
        self._hf_save(run)

    def _hf_save(self, run: _Run) -> None:
        """The run record of an hf run: the RECORD text, the builder's and
        every node's fingerprint and gate reports, and the slots."""
        hf = run.hf
        assert hf is not None
        text_path = os.path.join(run.run_dir, "hf_record.txt")
        if not os.path.exists(text_path):
            with open(text_path, "w") as f:
                f.write(hf["record_text"])
        with open(os.path.join(run.run_dir, "hf_admission.json"), "w") as f:
            json.dump(
                {
                    "hf_code": hf["hf_code"],
                    "builder": hf["builder"],
                    "fingerprints": hf["fingerprints"],
                    "reports": hf["reports"],
                    "slots": {f"{r},{p}": e for (r, p), e in hf["slots"].items()},
                },
                f,
                indent=1,
                default=str,
            )

    def _plan(self, run: _Run, job: Job, infos: list, hops, worst, nic, emus):
        """The planner over the selected workers (BA-48 inputs, BA-49 arming):
        -> (spec, rationale, groups[stage][peer] NodeInfos from the plan's
        placement). The zero-bubble schedule is solved for the placed slots,
        a pieces plan is re-checked under its re-weighted ownership (BA-47),
        and then every slot's fit is checked; a stage that does not fit
        re-plans at half the microbatch."""
        plan_job = dict(job.plan or {})
        self._sched_cap = {}  # node_id -> activation budget calibrated by measured fits (CONTROL-I26)
        self._recompute_on = set()  # node_ids a measured fit sent back to recomputing (RUNTIME-I25)
        plan_job.setdefault("tag", run.run_id)
        plan_job.setdefault("ckpt_every", self.cfg.default_ckpt_every)
        plan_job.setdefault("dial_timeout_s", self.cfg.dial_timeout_s)
        if not plan_job.get("device"):
            plan_job["device"] = infos[0].device
        link = (
            {"mbps": worst["mbps"], "rtt_ms": worst["rtt_ms"], "duplex": worst["duplex"]}
            if worst
            else {"mbps": 1_000_000.0, "rtt_ms": 0.0, "duplex": "per_direction"}
        )
        nodes = plan_fleet_nodes(infos, nic)
        run.pieces_nodes = {n.node_id: d for n, d in zip(infos, nodes)}
        run.pieces_s2_refuses = True
        mb_override = plan_job.get("mb_size")
        # arch 'hf' (CONTROL-I24, BA-424/425/461): the record is built and every
        # selected node and spare gated once, before the planner; the builder's
        # pins reach build_spec, and each plan's map is admitted before it is
        # profiled or fitted.
        hf_kw: dict = {}
        if _plan_model(plan_job)["arch"] == "hf":
            # Read from the plan dict as fleetplan.model_config reads it (the
            # preset, then the plan's non-None keys): model_config itself
            # refuses an hf plan until these pins exist (I-14).
            base = make_config(plan_job.get("model", "toy"))

            def pick(k):
                return getattr(base, k, None) if plan_job.get(k) is None else plan_job[k]

            pool = [self.workers[n.node_id] for n in infos] + self._class_spares(
                infos[0].device_class
            )
            out = self._hf_first_launch(
                run,
                # a fine-tune's config is its checkpoint's config.json (I-14)
                fleetplan.plan_hf_config(pick("hf_config"), pick("hf_model")),
                {"compute_dtype": pick("compute_dtype"), "seq_len": pick("seq_len")},
                int(plan_job.get("init_seed", JobSpec.init_seed)),
                int(mb_override or JobSpec.mb_size),
                pool,
            )
            hf_kw["hf_pins"] = {
                "hf_buffers": out["record"],
                "hf_census": out["census"],
                "hf_code": out["hf_code"],
                "fields": out["fields"],
            }
        for attempt in range(1, FIT_ATTEMPTS + 1):
            job_dict = dict(plan_job, **({"mb_size": mb_override} if mb_override else {}))
            fleet = {
                "run_id": run.run_id,
                "nodes": nodes,
                "link": link,
                "k_passes": {
                    "value": admission.counted_k(plan_job.get("adapt_budget_ms")),
                    "source": "counted",
                },
                "job": job_dict,
            }
            # Read at call time: arming is the module flag (tripwire pieces-unarmed).
            try:
                out = fleetplan.build_spec(fleet, allow_pieces=PIECES_ARMED, **hf_kw)
            except fleetplan.PlanRefused as e:
                missing = missing_measurements(nodes)
                if not missing:
                    raise
                # The planner's report lists precisions; name what was not measured.
                raise fleetplan.PlanRefused(
                    f"{e}\ninputs not measured: " + "; ".join(missing)
                ) from e
            spec = out["spec"]
            mismatch = plan_model_mismatch(job_dict, spec.model)
            if mismatch:
                raise RuntimeError(
                    "the planned spec's model differs from the plan (BA-461): "
                    + "; ".join(mismatch)
                )
            for w in out["warnings"]:
                self.log("warn", f"planner: {w}")
            self.log("info", "plan: " + out["rationale"]["decision"])
            groups = [[infos[p["node"]] for p in row] for row in out["placement"]]
            if run.hf is not None:
                # I-7's JobSpec.device_classes (stage-m4), set by name.
                setattr(spec, "device_classes", self._hf_admit_map(run, groups))  # noqa: B010
            links = placement.emulated_links(groups, emus if all(emus.values()) else None)
            if links is not None:
                spec.links = links
            schedule = None
            if spec.stages > 1:
                # The measured hops are in chain order: they are this plan's
                # hops only when it is that chain, one node per stage.
                chain = [n.node_id for g in groups for n in g] == [n.node_id for n in infos]
                schedule = self._zero_bubble(spec, groups, hops if chain else [])
                run.plan_hops = list(hops) if chain else []
            self._select_recompute(spec, groups)
            pieces_warnings = self._check_pieces(spec, groups, run)
            if not self.cfg.fit_check:
                break
            fits, why, over = self._fit_check(spec, groups)
            if fits:
                break
            kept = [
                n.node_id
                for r, g in enumerate(groups)
                for n in g
                if n.node_id in over and spec.recompute and not spec.recomputes(r)
            ]
            if kept and attempt < FIT_ATTEMPTS:
                # RUNTIME-I25 first: a stage that kept F's graph and overshot recomputes again, before any budget
                # is calibrated against a peak that graph inflated.
                self._recompute_on.update(kept)
                self.log("warn", f"{why}; the stage recomputes again (RUNTIME-I25)")
                continue
            fresh = [nid for nid in over if nid not in self._sched_cap]
            if fresh and attempt < FIT_ATTEMPTS:
                # The memory models are analytic; the fit is measured. First the schedule: each overflowing
                # node's activation budget is calibrated against what its pending microbatches really took
                # (calibrated_budget), so the zero-bubble solve holds fewer of them in flight there and the
                # cut (the load balance) stays as planned.
                notes = []
                for r, g in enumerate(groups):
                    for p, n in enumerate(g):
                        if n.node_id in fresh:
                            peak, budget = over[n.node_id]
                            cap = calibrated_budget(
                                spec, r, p, peak, budget, oom=peak >= n.memory_bytes()
                            )
                            self._sched_cap[n.node_id] = cap
                            notes.append(
                                f"rank {r} schedules with {cap / 2**30:.2f} GiB of activations"
                            )
                self.log("warn", f"{why}; re-solving the schedule: " + "; ".join(notes))
                continue
            if over and not plan_job.get("partition") and attempt < FIT_ATTEMPTS:
                # Then the cut: an MoE stage at one sample per microbatch has no microbatch left to halve, while
                # the cut can move its layers to a node with room.
                notes = shrink_for_overshoot(nodes, [n.node_id for n in infos], over)
                self.log("warn", f"{why}; re-planning the cut: " + "; ".join(notes))
                continue
            nxt = (mb_override or spec.mb_size) // 2
            if attempt == FIT_ATTEMPTS or nxt < 1:
                raise RuntimeError(
                    f"the planned stage does not fit and shrinking the "
                    f"microbatch did not help: {why}"
                )
            self.log("warn", f"{why}; re-planning at mb_size={nxt}")
            mb_override = nxt
        chosen = out["rationale"].get("chosen")
        rationale = dict(
            out["rationale"],
            report=out["report"],
            placement=[[f"{n.name}({n.device}/{n.gpu_name})" for n in g] for g in groups],
            emulated_links=spec.links if any(lk.get("bw_mbps") for lk in spec.links) else None,
            inputs=plan_inputs(nodes, fleet, "measured" if worst else "default"),
            plan_placement=out["placement"],
            stage_widths=out["stage_widths"],
            shard=out["shard"],
            wire=out["wire"],
            chosen_step_s=next(
                (c["step_s"] for c in out["candidates"] if c["name"] == chosen), None
            ),
            plan_candidates=[
                {
                    **{k: c[k] for k in ("name", "S", "w", "shard", "wire", "admitted", "step_s")},
                    "reasons": [{"kind": r["kind"], "text": r["text"]} for r in c["reasons"]],
                }
                for c in out["candidates"]
            ],
            pieces_warnings=pieces_warnings,
        )
        if schedule is not None:
            rationale["schedule"] = schedule
        return spec, rationale, groups

    def _profile(self, spec: JobSpec, groups) -> list:
        """F/B/W seconds of every slot, measured on the node that runs it.
        -> profiles[stage][peer]."""
        body = JobSpec.from_json(spec.to_json()).to_json().encode()
        profiles: list = [[None] * len(g) for g in groups]
        errs: list = []

        def one(r, p):
            w = self.workers[groups[r][p].node_id]
            rep = self._request(
                w, {"kind": "PROFILE", "rank": r, "peer": p, "reps": self.cfg.profile_reps}, body
            )
            prof = rep.get("profile") or {}
            if not prof.get("ok"):
                raise RuntimeError(f"profile of slot ({r}, {p}) failed: {prof.get('error')}")
            profiles[r][p] = prof

        ts = [
            threading.Thread(target=lambda r=r, p=p: _catch(errs, one, r, p))
            for r, g in enumerate(groups)
            for p in range(len(g))
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise errs[0]
        return profiles

    def _refine_partition(self, spec: JobSpec, groups, profiles) -> bool:
        """Re-cut the layers by each stage's measured throughput (cost units
        per second of F+B+W, summed over the stage's peers, on their own
        nodes) instead of a matmul benchmark; applied only when it shortens
        the predicted slowest stage by more than 5%."""
        res = refined_partition(spec, profiles, _stage_fits(spec, groups), min_gain=0.05)
        if res is None:
            return False
        new, before, after = res
        self.log(
            "info",
            f"partition refined by measured stage throughput: {spec.partition} -> "
            f"{new} (slowest stage {before * 1e3:.1f} -> ~{after * 1e3:.1f} ms "
            f"per microbatch)",
        )
        spec.partition = new
        spec.reschedule()
        return True

    def _zero_bubble(self, spec: JobSpec, groups, hops, refine: bool = False) -> dict:
        """Profile every slot, refine the cut, solve and pin the zero-bubble
        schedule. Without profiles the spec keeps the schedule finalize()
        solved from the analytic cost model: still zero-bubble, less tuned."""
        from wantrain.scheduler.zero_bubble import schedule_for_spec

        refined = False
        try:
            profiles = self._profile(spec, groups)
            refined = refine and self._refine_partition(spec, groups, profiles)
            if refined:
                profiles = self._profile(spec, groups)
        except (RuntimeError, WorkerLost) as e:
            self.log("warn", f"profiling failed ({e}); keeping the analytic zero-bubble schedule")
            return {"source": "analytic", "reason": str(e)}
        out = schedule_for_spec(
            spec,
            profiles,
            hops if len(hops) else None,
            _cap_budgets(_mem_budgets(spec, groups), groups, self._sched_cap),
            self.cfg.zb_mem_mode,
        )
        sched, cfg = out["schedule"], out["config"]
        info = {
            "source": "profiled",
            "costs_ms": [
                [round(x * 1e3, 3) for x in (cfg.cost_f[r], cfg.cost_b[r], cfg.cost_w[r])]
                for r in range(cfg.n_stages)
            ],
            "comm_ms": [round(c * 1e3, 3) for c in cfg.comm],
            "step_s": round(out["slot_makespan"], 4),
            "bubble": round(sched.bubble_rate, 4),
            "params": sched.params,
            "peer_weights": [[round(w, 3) for w in row] for row in out["peer_weights"]],
            "partition_refined": bool(refined),
        }
        self.log(
            "info",
            f"zero-bubble schedule: predicted {out['slot_makespan']:.3f} s/step, bubble "
            f"{sched.bubble_rate:.1%}; F/B/W ms per stage {info['costs_ms']}, hop comm ms "
            f"{info['comm_ms']}, peer weights {info['peer_weights']}",
        )
        return info

    def _select_recompute(self, spec: JobSpec, groups) -> None:
        """RUNTIME-I25: set spec.recompute_stages from the solved orders (zero_bubble.selective_recompute), the
        stages a measured fit sent back to recomputing excluded. WANTRAIN_SELECTIVE_RECOMPUTE=off keeps every
        stage recomputing."""
        from wantrain.scheduler.zero_bubble import selective_recompute

        if os.environ.get("WANTRAIN_SELECTIVE_RECOMPUTE", "off") != "on":
            return
        force = {r for r, g in enumerate(groups) if any(n.node_id in self._recompute_on for n in g)}
        stages = selective_recompute(spec, force)
        spec.recompute_stages = stages
        if stages is not None:
            keep = [r for r, x in enumerate(stages) if not x]
            self.log(
                "info",
                f"recompute: stage(s) {keep} keep F's graph (one microbatch between F and B)",
            )

    def _fit_check(self, spec: JobSpec, groups):
        body = spec.to_json().encode()
        bad: list[tuple[int, str]] = []
        unknown: list[tuple[int, str]] = []
        over: dict = {}  # node_id -> (peak, budget) bytes, for every measured overshoot
        slots = [(r, p, n) for r, g in enumerate(groups) for p, n in enumerate(g)]

        def ask(slot):
            r, p, n = slot
            return self._request(
                self.workers[n.node_id], {"kind": "FIT", "rank": r, "peer": p}, body
            )

        # Each slot measures its own GPU: ask every node at once (one round of the slowest build, not their sum),
        # then judge in rank order, so the log and the first refusal read as before.
        with ThreadPoolExecutor(max_workers=max(1, len(slots))) as pool:
            reps = list(pool.map(ask, slots))
        for (r, _p, n), rep in zip(slots, reps):
            fit = rep.get("fit") or {}
            if fit.get("skipped"):
                continue
            budget = n.memory_bytes() * (1 - FIT_HEADROOM)
            if not fit.get("ok"):
                self.log("warn", f"fit rank {r}: {str(fit.get('error'))[:160]}")
                (bad if fit.get("oom") else unknown).append((r, str(fit.get("error"))))
                if fit.get("oom") and n.memory_bytes():
                    # out of memory before a peak was read: the demand was at least the whole card
                    over[n.node_id] = (n.memory_bytes(), budget)
                continue
            peak = fit.get("peakReservedBytes") or 0
            if peak:
                self._fit_peaks[n.node_id] = (tuple(spec.partition[r]), int(peak), int(budget))
            self.log(
                "info",
                f"fit rank {r}: peak {peak / 2**30:.2f} GiB of {budget / 2**30:.2f} GiB budget",
            )
            if peak > budget:
                bad.append((r, f"peak {peak / 2**30:.2f} GiB > budget {budget / 2**30:.2f}"))
                over[n.node_id] = (peak, budget)
        if bad:
            return False, f"rank {bad[0][0]} does not fit: {bad[0][1]}", over
        return True, "", over

    # ================================================================ launch
    def _launch(
        self, run: _Run, groups=None, spec_json: str | None = None, start_step: int = 0
    ) -> None:
        spec_json = spec_json or run.spec_json
        spec = JobSpec.from_json(spec_json)
        final_rank = spec.stages - 1
        with run.lock:
            run.attempt += 1
            now = time.time()
            run.states = {
                slot: SlotState(
                    slot=slot,
                    node_id=nid,
                    launched=True,
                    last_heartbeat=now,
                    started_at=now,
                    start_step=start_step,
                )
                for slot, nid in run.slots.items()
            }
            run.results = {}
            # The window holds only this attempt's summaries of its current
            # owners (BA-23); an older attempt's are refused by _slot_for.
            run.telem_window = {}
            run.slot_base, run.slot_hist = {}, {}  # CONTROL-I27 baselines are per attempt
            run.replan_checked = start_step - 1
        # Parts staged under an earlier attempt can never be committed now.
        run.store.drop_staged()
        self.log(
            "info",
            f"launching attempt {run.attempt} (resume={spec.resume}) on {len(run.slots)} slot(s)",
        )
        # Last rank first, so every listener is up before its upstream dials.
        for slot in sorted(run.slots, reverse=True):
            w = self.workers[run.slots[slot]]
            self._request(
                w,
                {
                    "kind": "ASSIGN",
                    "run_id": run.run_id,
                    "attempt": run.attempt,
                    "rank": slot[0],
                    "peer": slot[1],
                    "final": slot[0] == final_rank and slot[1] == 0,
                },
                spec_json.encode(),
                timeout=60,
            )
        self._sync_all_connections(run)

    def _sync_all_connections(self, run: _Run) -> None:
        for nid in set(run.slots.values()):
            self._sync_slot_connection(self.workers[nid])

    # ================================================================ monitor
    @staticmethod
    def _final_step(run: _Run) -> int | None:
        if run.spec is not None:
            return run.spec.steps - 1
        return JobSpec.from_json(run.spec_json).steps - 1 if run.spec_json else None

    def _judge(self, run: _Run) -> LivenessJudge:
        c = self.cfg
        return LivenessJudge(
            heartbeat_timeout_s=c.heartbeat_timeout_s,
            reconnect_grace_s=c.reconnect_grace_s,
            stall_floor_s=c.stall_floor_s,
            stall_steps=c.stall_steps,
            strike_gap_s=c.strike_gap_s,
            first_step_timeout_s=c.first_step_timeout_s,
            exit_report_grace_s=c.exit_report_grace_s,
            final_step=self._final_step(run),
            shutdown_limit_s=c.shutdown_limit_s,
        )

    def _monitor(self, run: _Run) -> RunResult:
        judge = self._judge(run)
        t_start = time.time()
        last_note = ""
        while True:
            time.sleep(self.cfg.poll_s)
            with run.lock:
                for slot, nid in run.slots.items():
                    w = self.workers[nid]
                    st = run.states[slot]
                    st.last_heartbeat = max(st.last_heartbeat, w.last_seen())
                    st.connected = w.connected
                    st.disconnected_at = w.disconnected_at
                if run.restore_t0 is not None and all(st.progress for st in run.states.values()):
                    run.restart_cost_s = time.time() - run.restore_t0
                    run.restore_t0 = None
                    self.log("info", f"restore to first step took {run.restart_cost_s:.2f} s")
                v = judge.judge(list(run.states.values()))
            if v.kind == "done":
                return self._complete(run)
            if self.cfg.max_run_s and time.time() - t_start > self.cfg.max_run_s:
                raise RuntimeError(f"run exceeded its {self.cfg.max_run_s:.0f}s budget")
            if v.kind in ("ok",):
                # Only here, between judge passes, on the thread that owns heal
                # and restart (BA-45): a convict or stall of this pass went to
                # its branch, and its new attempt starts a new window.
                if self._watch_slow(run) or self._maybe_rebalance(run):
                    judge = self._judge(run)
                    last_note = ""
                continue
            if v.kind == "suspect":
                if v.why != last_note:
                    self.log("warn", f"suspect {v.slot}: {v.why}")
                    last_note = v.why
                continue
            if v.kind == "failed":
                raise RuntimeError(v.why)
            if v.kind == "convict":
                self.log("error", f"slot {v.slot} convicted: {v.why}")
                self._heal(run, v.slot, v.why)
                judge = self._judge(run)
                last_note = ""
                continue
            if v.kind == "stalled":
                self.log("error", f"fleet stalled: {v.why}")
                self._restart_in_place(run, v.why)
                judge = self._judge(run)
                last_note = ""
                continue

    def _complete(self, run: _Run) -> RunResult:
        spec = JobSpec.from_json(run.spec_json)
        final = (spec.stages - 1, 0)
        deadline = time.time() + 60
        # Every slot sends its exit-time TELEMETRY file before its RESULT on the
        # same channel; waiting for every RESULT (same bound) keeps a late file
        # from arriving after the run is closed and being discarded.
        while not all(s in run.results for s in run.slots) and time.time() < deadline:
            time.sleep(0.1)
        res = run.results.get(final)
        if res is None or "losses" not in res:
            return self._finish(
                run, False, "training finished but the final stage's losses never arrived"
            )
        with open(os.path.join(run.run_dir, "losses.json"), "w") as f:
            json.dump(res["losses"], f, indent=2)
        self.log(
            "info",
            f"run {run.run_id} complete: {len(res['losses']['losses'])} losses, "
            f"{run.attempt} attempt(s), {len(run.heals)} heal(s), "
            f"{run.restarts} restart(s), {len(run.replans)} rebalance(s)",
        )
        return self._finish(run, True, None, res["losses"])

    def _finish(self, run: _Run, ok: bool, reason, losses=None) -> RunResult:
        for nid in set(run.slots.values()):
            w = self.workers.get(nid)
            if w is None:
                continue
            if w.connected and not w.excluded:
                try:
                    self._request(w, {"kind": "RELEASE", "run_id": run.run_id}, timeout=30)
                except (WorkerLost, RuntimeError):
                    pass
            w.run_id = None
        for w in self.workers.values():
            if w.run_id == run.run_id:
                w.run_id = None
        result = RunResult(
            ok=ok,
            run_id=run.run_id,
            reason=reason,
            losses=losses,
            run_dir=run.run_dir,
            attempts=run.attempt,
            heals=list(run.heals),
            restarts=run.restarts,
            rationale=run.rationale,
            events=list(self.events),
            replans=list(run.replans),
        )
        with open(os.path.join(run.run_dir, "result.json"), "w") as f:
            json.dump(
                {
                    "ok": ok,
                    "reason": reason,
                    "attempts": run.attempt,
                    "heals": run.heals,
                    "restarts": run.restarts,
                    "replans": run.replans,
                    "events": [list(e) for e in self.events],
                },
                f,
                indent=2,
                default=str,
            )
        self.run = None
        return result

    # ================================================================ recovery
    def _abort_all(self, run: _Run) -> None:
        def one(nid):
            w = self.workers.get(nid)
            if w and w.connected:
                try:
                    self._request(w, {"kind": "ABORT", "run_id": run.run_id}, timeout=60)
                except (WorkerLost, RuntimeError) as e:
                    self.log("warn", f"abort on {w.info.name}: {e}")

        ts = [threading.Thread(target=one, args=(nid,)) for nid in set(run.slots.values())]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    def _restore_and_relaunch(self, run: _Run, step, spec_json: str) -> None:
        spec = JobSpec.from_json(spec_json)
        # R (CONTROL-I21) is measured from here to the attempt's first step,
        # from step 0 too: a stale t0 of an earlier relaunch overstates it.
        t0 = time.time()
        if step is not None:
            widths = spec.stage_widths or [1] * spec.stages

            def push(slot):
                # The stored wave as a unit: a RESTORE_PART per listed file,
                # then the RESTORE with the manifest, on the same connection.
                # A slot of a wide pieces stage resumes from the stage's
                # logical state, read from every old slot's wave (RUNTIME-I12),
                # so it gets its stage-mates' waves too, its own last.
                w = self.workers[run.slots[slot]]
                r = slot[0]
                srcs = [slot]
                if spec.shard == "pieces" and widths[r] > 1:
                    srcs = [(r, k) for k in range(widths[r]) if k != slot[1]] + [slot]
                for src in srcs:
                    d, m = run.store.wave_path(src, step), run.store.manifest(src, step)
                    head = {"run_id": run.run_id, "rank": src[0], "peer": src[1], "step": step}
                    ch = w.ch
                    if not w.connected or ch is None:
                        raise WorkerLost(w.node_id, "not connected")
                    try:
                        for e in m["files"]:
                            ch.send_file(
                                dict(head, kind="RESTORE_PART", name=e["name"]),
                                os.path.join(d, e["name"]),
                            )
                    except ChannelClosed as e:
                        raise WorkerLost(w.node_id, f"send failed ({e})") from e
                    rep = self._request(
                        w,
                        dict(head, kind="RESTORE", tag=spec.tag, manifest=m),
                        timeout=1800,
                    )
                    if not rep.get("ok"):
                        raise RuntimeError(
                            f"shard restore to {w.info.name} failed: {rep.get('error')}"
                        )

            errs = []
            ts = [threading.Thread(target=lambda s=s: _catch(errs, push, s)) for s in run.slots]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            if errs:
                raise errs[0]
            self.log("info", f"wave {step} restored to every slot (sha256 verified)")
        run.restore_t0 = t0
        # A resumed attempt's first optimizer step is the one after the wave.
        self._launch(run, spec_json=spec_json, start_step=0 if step is None else step + 1)

    def _slot_peers(self, run: _Run) -> list:
        """slot_peers[stage][peer]: the key each slot's node proved in its
        control handshake (CONTROL-I22)."""
        rows: dict = {}
        for (r, p), nid in run.slots.items():
            pid = self.workers[nid].peer_id
            if pid is None:
                raise RuntimeError(f"slot {(r, p)}: node {nid} proved no key in its handshake")
            rows.setdefault(r, {})[p] = pid
        return [[rows[r][p] for p in sorted(rows[r])] for r in sorted(rows)]

    def _resume_spec(self, run: _Run, step):
        # CONTROL-I25 at every admission: heal, restart in place and rebalance
        # all build their map here; nothing relaunches on a mixed one.
        mixed = mixed_fleet(
            (self.workers[nid].info.name, self.workers[nid].info.device_class)
            for _, nid in sorted(run.slots.items())
        )
        if mixed is not None:
            raise RuntimeError(f"admission refused (CONTROL-I25); nothing relaunched: {mixed}")
        if run.hf is not None:
            # CONTROL-I24 at every admission (BA-426): heal, restart in place
            # and rebalance all build their map here.
            self._hf_readmit(run)
        spec = JobSpec.from_json(run.spec_json)
        groups = {}
        for (r, p), nid in run.slots.items():
            groups.setdefault(r, {})[p] = self.workers[nid].info
        ordered = [[groups[r][p] for p in sorted(groups[r])] for r in sorted(groups)]
        endpoints = [[[n.host, n.data_port, n.bind_port] for n in g] for g in ordered]
        devices = [[n.device for n in g] for g in ordered]
        slot_peers = self._slot_peers(run)
        devs = (
            devices
            if (spec.devices is not None or any(d != spec.device for row in devices for d in row))
            else None
        )
        if step is None:
            d = json.loads(run.spec_json)
            d["endpoints"] = endpoints
            if devs is not None:
                d["devices"] = devs
            d.update(slot_peer_pins(slot_peers))
            if run.ownership is not None:
                d.update(run.ownership)
            d["resume"] = False
            return json.dumps(d, indent=2)
        return patch_spec_for_resume(
            run.spec_json,
            endpoints,
            devs,
            slot_peers=slot_peers,
            resume_step=step,
            ownership=run.ownership,
        )

    def _heal(self, run: _Run, slot, why: str) -> None:
        if len(run.heals) >= self.cfg.max_heals:
            raise RuntimeError(
                f"slot {slot} died ({why}) and the heal budget ({self.cfg.max_heals}) is spent"
            )
        dead_w = self.workers[run.slots[slot]]
        self._exclude(dead_w.node_id, why, run.run_id)
        self._abort_all(run)
        survivors = [self.workers[n].info for s, n in run.slots.items() if s != slot]
        # CONTROL-I25: only a spare of the run's device class; spares of another
        # class stay registered and are never assigned.
        run_class = dead_w.info.device_class
        idle = [w.info for w in self.idle_workers()]
        spares = [w.info for w in self._class_spares(run_class, dead_w.info.gpu_name)]
        if not run_class or not spares:
            raise RuntimeError(
                f"slot {slot} died ({why}) and the heal fails (CONTROL-I25); nothing "
                f"relaunched: MIXED-FLEET: no spare of the run's device class "
                f"{run_class or '<none>'}"
                + (
                    f" and the slot's GPU model {dead_w.info.gpu_name}"
                    if run_class.startswith("cuda:sm_")
                    else ""
                )
                + f" (pool: {class_counts(idle)})"
            )
        # Read from the JSON, not JobSpec fields, so the rule holds whichever
        # spec.py is on the tree (interface I-9).
        d = json.loads(run.spec_json)
        widths = d.get("stage_widths") or [1] * d["stages"]
        # CONTROL-I9 (BA-30, BA-31): the peers of a pieces stage hold their
        # bits on one device class, so its slot is healed only by the identical
        # probed model; every other slot keeps the configured match.
        pieces_slot = d.get("shard", "zero1") == "pieces" and widths[slot[0]] > 1
        match = "model" if pieces_slot else self.cfg.replacement_match
        # CONTROL-I24 (BA-426): an hf slot takes only a spare whose gate, run
        # now, reports the slot's recorded device class; the readmission in
        # _resume_spec then checks the map without gating it again.
        device_class, gate_classes = None, None
        if run.hf is not None:
            gated = self._hf_gate(run, [self.workers[s.node_id] for s in spares], tolerate=True)
            run.hf["fresh"] = gated
            device_class = run.hf["slots"][slot]["device_class"]
            gate_classes = {
                n: [
                    x["device_class"]
                    for x in run.hf["reports"][n]
                    if isinstance(x, dict) and x.get("ok") is True and x.get("device_class")
                ]
                for n in gated
            }
        pick = find_replacement(
            spares,
            dead_w.info,
            survivors,
            GPTConfig(**d["model"]).compute_dtype,
            match,
            self.memory.dead_keys(),
            self.cfg.selection.distinct_hosts,
            probed_class=pieces_slot,
            device_class=device_class,
            gate_classes=gate_classes,
        )
        if pick is None:
            need = (
                f"the identical probed device model {dead_w.info.gpu_name!r} on "
                f"{dead_w.info.device!r} (a slot of a pieces stage; a declared class "
                "never matches)"
                if pieces_slot
                else f"matching '{match}'"
            )
            if device_class is not None:
                need += f", whose gate reports the slot's device class {device_class!r}"
            raise RuntimeError(
                f"slot {slot} died ({why}) and no registered spare can take "
                f"its place (needs >= {dead_w.info.memory_bytes() / 2**30:.1f} "
                f"GB on a distinct host, {need})"
            )
        new, note = pick
        nw = self.workers[new.node_id]
        nw.run_id = run.run_id
        run.slots[slot] = new.node_id
        step = run.store.common_step(list(run.slots))
        run.heals.append(
            {
                "slot": list(slot),
                "at": time.time(),
                "reason": why,
                "from": dead_w.info.name,
                "to": new.name,
                "resumed_from_step": step,
            }
        )
        self.log("info", f"replacement for slot {slot}: {new.name} -- {note}")
        if step is None:
            self.log(
                "warn",
                "no checkpoint wave is held for every slot: restarting from "
                "step 0 (deterministic, only time is lost)",
            )
        else:
            self.log("info", f"every slot resumes after step {step} under the original spec")
        spec_json = self._resume_spec(run, step)
        self._restore_and_relaunch(run, step, spec_json)

    def _restart_in_place(self, run: _Run, why: str) -> None:
        if run.restarts >= self.cfg.max_restarts:
            raise RuntimeError(
                f"fleet cannot progress ({why}) and the restart budget "
                f"({self.cfg.max_restarts}) is spent"
            )
        run.restarts += 1
        self._abort_all(run)
        step = run.store.common_step(list(run.slots))
        self.log("info", f"restarting in place from step {step} (restart {run.restarts})")
        self._restore_and_relaunch(run, step, self._resume_spec(run, step))

    # ================================================================ rebalance
    def _owned_spec(self, run: _Run) -> JobSpec:
        """The current attempts' recipe: the launch spec under the ownership the
        last rebalance applied."""
        d = json.loads(run.spec_json)
        if run.ownership is not None:
            d.update(run.ownership)
        return JobSpec.from_json(json.dumps(d)).finalize()

    def _maybe_rebalance(self, run: _Run) -> bool:
        """Evaluate the rebalance rule (CONTROL-I21, SPEC-control task 2) once
        per new common wave, from the current attempt's window, and apply it
        when it holds. Called only by _monitor (BA-45). True when a new
        attempt was launched."""
        from wantrain.scheduler.zero_bubble import replan

        if len(run.replans) >= self.cfg.max_replans:
            return False
        step = run.store.common_step(list(run.slots))
        final = self._final_step(run)
        if step is None or step <= run.replan_checked or final is None or step >= final:
            return False
        with run.lock:
            windows = {slot: list(run.telem_window.get(slot, ())) for slot in run.slots}
        if any(len(w) < REPLAN_MIN_SUMMARIES for w in windows.values()):
            return False
        run.replan_checked = step
        spec = self._owned_spec(run)
        if self._maybe_recut(run, spec, windows, step, final):
            return True
        infos = [
            [self.workers[run.slots[(r, p)]].info for p in range(spec.stage_widths[r])]
            for r in range(spec.stages)
        ]
        try:
            plan = replan(
                spec,
                windows,
                REPLAN_MIN_GAIN,
                _cap_budgets(_mem_budgets(spec, infos), infos, self._sched_cap),
                self.cfg.zb_mem_mode,
            )
        except ValueError as e:
            self.log("warn", f"rebalance not evaluated at wave {step}: {e}")
            return False
        cost = run.restart_cost_s
        cost = self.cfg.replan_restart_cost_s if cost is None else cost
        ok, why = replan_rule(plan, final - step, cost, len(run.replans), self.cfg.max_replans)
        if plan["decision"]:
            self.log("info", f"rebalance at wave {step}: {why}")
        if not ok:
            return False
        refused = self._pieces_rebalance_refusals(run, spec, plan)
        if refused:
            texts = [r.text() for r in refused]
            self.log(
                "warn",
                f"rebalance at wave {step} not applied: SCHEDULER-I16 refuses the new "
                f"ownership ({'; '.join(texts)}); the run continues on its current plan",
            )
            # Recorded with the replans (it spends one of max_replans, so a
            # refused ownership is not re-solved at every later wave).
            run.replans.append({"step": step, "applied": False, "refused": texts})
            return False
        self._rebalance(run, plan)
        return True

    def _pieces_rebalance_refusals(self, run: _Run, spec: JobSpec, plan: dict) -> list:
        """BA-44 (SCHEDULER-I16 'at every re-plan'): before a rebalance of a
        run with a pieces stage is applied, its slots' class and the memory
        and S2 terms are re-checked under the new mb_owner and orders, on the
        inputs the run was admitted on. -> the refusals (empty: apply)."""
        if spec.shard != "pieces":
            return []
        d = json.loads(spec.to_json())
        d.update({k: plan[k] for k in OWNERSHIP_FIELDS})
        new = JobSpec.from_json(json.dumps(d)).finalize()
        groups = [
            [self.workers[run.slots[(r, p)]].info for p in range(new.stage_widths[r])]
            for r in range(new.stages)
        ]
        flat = [n for g in groups for n in g]
        # a healed slot's node was not in the launch fleet: its inputs as probed
        nodes = {**_nodes_by_id(flat, None), **run.pieces_nodes}
        refusals, _warnings = pieces_admission(new, groups, nodes, run.pieces_s2_refuses)
        return refusals

    def _rebalance(self, run: _Run, plan: dict) -> None:
        """Apply a re-plan as its own attempt (BA-24): abort, re-read the
        common wave (ABORT ships newer waves and the intake's prune may drop
        the decided one), restore it and relaunch under the new ownership. It
        consumes neither max_restarts nor max_heals."""
        self._abort_all(run)
        step = run.store.common_step(list(run.slots))
        if step is None:
            self.log("error", "REPLAN abandoned: no wave is held for every slot after the abort")
            raise RuntimeError(
                "rebalance abandoned: no wave is held for every slot after the abort "
                "(a rebalance never restarts from step 0)"
            )
        run.ownership = {k: plan[k] for k in OWNERSHIP_FIELDS}
        rec = {
            "step": step,
            "pred_cur_s": plan["pred_cur_s"],
            "pred_new_s": plan["pred_new_s"],
            "owner_changed": plan["owner_changed"],
            "peer_weights": plan["peer_weights"],
        }
        run.replans.append(rec)
        self.log("info", f"REPLAN {json.dumps(rec, separators=(',', ':'))}")
        self._restore_and_relaunch(run, step, self._resume_spec(run, step))

    # ---------------- RUNTIME-I26 / CONTROL-I27: re-cut and slow stages ----------------

    def _maybe_recut(
        self, run: _Run, spec: JobSpec, windows, step, final, why: str | None = None
    ) -> bool:
        """Re-cut the chain from the window's measured stage costs (refined_partition) when the slowest stage
        shrinks by cfg.recut_min_gain and the saving over the remaining steps beats cfg.recut_cost_s. `why`: a slow
        stage forced the evaluation (CONTROL-I27). True when a new attempt was launched."""
        if not self.cfg.recut or os.environ.get("WANTRAIN_RECUT", "on") == "off":
            return False
        refusal = recut_refusal(spec)
        if refusal is not None or len(run.replans) >= self.cfg.max_replans or step is None:
            return False
        from wantrain.scheduler.zero_bubble import profiles_from_telemetry

        try:
            profiles = profiles_from_telemetry(spec, windows)
        except ValueError as e:
            self.log("warn", f"re-cut not evaluated at wave {step}: {e}")
            return False
        infos = [[self.workers[run.slots[(r, 0)]].info] for r in range(spec.stages)]
        res = refined_partition(
            spec, profiles, _stage_fits(spec, infos, self._fit_peaks), self.cfg.recut_min_gain
        )
        if res is None:
            return False
        new, before, after = res
        saving = (before - after) * spec.microbatches * max(0, final - step)
        # WANTRAIN_RECUT_COST_S overrides the cost an operator expects a re-cut to take (a small test model).
        cost = float(os.environ.get("WANTRAIN_RECUT_COST_S", self.cfg.recut_cost_s))
        note = (
            f"re-cut {spec.partition} -> {new}: slowest stage {before * 1e3:.0f} -> ~{after * 1e3:.0f} ms per "
            f"microbatch, ~{saving:.0f} s saved over {final - step} steps against {cost:.0f} s"
            + (f" ({why})" if why else "")
        )
        if saving <= cost:
            self.log("info", f"{note}: not worth it")
            return False
        self.log("info", note)
        return self._recut(run, new, note)

    def _recut(self, run: _Run, new_part: list, note: str) -> bool:
        """Apply a re-cut as its own attempt: abort, re-solve and fit the new cut on the same nodes, convert the
        common wave of every slot (runtime/recut.py) under a new tag, drop the newer waves, restore and relaunch.
        A new cut that does not fit leaves the run on its current one."""
        import torch

        from wantrain.fault_tolerance.checkpoint import WAVE_STATE
        from wantrain.runtime.recut import entry_param_names, recut_states

        self._abort_all(run)
        step = run.store.common_step(list(run.slots))
        if step is None:
            self.log("error", "RECUT abandoned: no wave is held for every slot after the abort")
            raise RuntimeError("re-cut abandoned: no wave is held for every slot after the abort")
        old = self._owned_spec(run)
        d = json.loads(run.spec_json)
        cuts = sum(1 for r in run.replans if r.get("recut"))
        d.update(
            partition=[list(x) for x in new_part],
            tag=f"{d['tag'].split('.cut')[0]}.cut{cuts + 1}",
            schedule_order=None,
            mb_owner=None,
            peer_weights=None,
            recompute_stages=None,
        )
        new = JobSpec.from_json(json.dumps(d)).finalize()
        groups = [[self.workers[run.slots[(r, 0)]].info] for r in range(new.stages)]
        if new.stages > 1:
            self._zero_bubble(new, groups, run.plan_hops)
        self._select_recompute(new, groups)
        if self.cfg.fit_check:
            fits, why, _over = self._fit_check(new, groups)
            if not fits:
                self.log(
                    "warn",
                    f"re-cut to {new_part} refused by the fit ({why}); the run continues on its cut",
                )
                run.replans.append({"step": step, "recut": False, "refused": why})
                self._restore_and_relaunch(run, step, self._resume_spec(run, step))
                return True
        names = entry_param_names(old.model)
        states = [
            torch.load(
                os.path.join(run.store.wave_path((r, 0), step), WAVE_STATE),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for r in range(old.stages)
        ]
        for r, payload in enumerate(recut_states(states, old.partition, new.partition, names)):
            run.store.replace_wave((r, 0), step, payload)
        del states
        run.store.drop_after(list(run.slots), step)
        run.spec, run.spec_json = new, new.to_json()
        run.ownership = None
        rec = {
            "step": step,
            "recut": True,
            "from": old.partition,
            "to": new.partition,
            "tag": new.tag,
        }
        run.replans.append(rec)
        self.log("info", f"RECUT {json.dumps(rec, separators=(',', ':'))}: {note}")
        self._restore_and_relaunch(run, step, self._resume_spec(run, step))
        return True

    def _watch_slow(self, run: _Run) -> bool:
        """CONTROL-I27: each slot's compute per microbatch (F + B + W medians of its summaries) against its own
        baseline, the median of its first cfg.slow_runs summaries after the warm-up steps; a slot above
        cfg.slow_ratio x baseline for cfg.slow_runs summaries in a row is replaced by a spare of the run's class
        (heal), or, with none, re-cut away from. True when a new attempt was launched."""
        with run.lock:
            windows = {slot: list(run.telem_window.get(slot, ())) for slot in run.slots}
        for slot, win in windows.items():
            # The attempt's first two summaries hold the warm-up and the one-time pinning of host moments.
            hist = [stage_compute_ms(s) for s in win[2:]]
            if len(hist) >= self.cfg.slow_runs and slot not in run.slot_base:
                run.slot_base[slot] = statistics.median(hist[: self.cfg.slow_runs])
            run.slot_hist[slot] = hist
        slow = slow_slots(run.slot_hist, run.slot_base, self.cfg.slow_ratio, self.cfg.slow_runs)
        if not slow:
            return False
        slot, ratio = max(slow, key=lambda x: x[1])
        why = f"slot {slot} computes {ratio:.2f}x its baseline per microbatch (CONTROL-I27)"
        w = self.workers[run.slots[slot]]
        spares = (
            self._class_spares(w.info.device_class, w.info.gpu_name) if w.info.device_class else []
        )
        step = run.store.common_step(list(run.slots))
        final = self._final_step(run)
        if step is None or final is None or step >= final:
            if run.slow_noted != slot:
                run.slow_noted = slot
                self.log("warn", f"{why}: no wave to resume from yet; acting at the next one")
            return False
        if spares and len(run.heals) < self.cfg.max_heals:
            self.log("warn", f"{why}: replaced by a spare")
            run.slot_base, run.slot_hist, run.slow_noted = {}, {}, None
            self._heal(run, slot, why)
            return True
        if run.slow_noted == ("recut", slot, step):
            return False
        run.slow_noted = ("recut", slot, step)
        self.log("warn", f"{why}: no spare; evaluating a re-cut on the recent costs")
        spec = self._owned_spec(run)
        recent = {s: win[-self.cfg.slow_runs :] for s, win in windows.items()}
        if self._maybe_recut(run, spec, recent, step, final, why):
            run.slot_base, run.slot_hist, run.slow_noted = {}, {}, None
            return True
        return False


def m4_unarmed_reason(model: dict) -> str | None:
    """Tripwire m4-unarmed (I-16, BA-427, BA-464): the refusal of a model dict
    ({'arch', 'moe_experts', ...}) whose arch is 'hf' or whose moe_experts > 0
    while M4_ARMED is False, else None. Pure; run_job, `wantrain local` and
    `wantrain stage` call it. A moe_experts that is not an int is refused too:
    the tripwire fails closed."""
    if M4_ARMED:
        return None
    if model.get("arch") == "hf":
        return "arch 'hf' is not armed: every leg of the m4 tripwire must land first"
    experts = model.get("moe_experts") or 0
    if isinstance(experts, bool) or not isinstance(experts, int) or experts > 0:
        return "moe_experts > 0 is not armed: every leg of the m4 tripwire must land first"
    return None


def _plan_model(plan: dict) -> dict:
    """The arch and moe_experts a plan-mode job declares, read as
    fleetplan.model_config reads them: the preset, then every key the plan
    gives that is not None."""
    preset = plan.get("model", "toy")
    base = PRESETS.get(preset, {}) if isinstance(preset, str) else {}
    model = {"arch": base.get("arch"), "moe_experts": base.get("moe_experts")}
    for k in model:
        if plan.get(k) is not None:
            model[k] = plan[k]
    return model


def unarmed_reason(job: Job) -> str | None:
    """Why run_job refuses this job, or None: tripwire m4-unarmed (always
    checked) and, while PIECES_ARMED is False, tripwire pieces-unarmed. A
    spec-mode job is read from its spec, a plan-mode job (no spec) from its
    plan dict; the fields are read without the JobSpec and GPTConfig fields
    existing (I-9, I-16)."""
    if job.spec is not None:
        m = job.spec.model
        model = {"arch": getattr(m, "arch", None), "moe_experts": getattr(m, "moe_experts", 0)}
        shard = getattr(job.spec, "shard", "zero1")
        wire = getattr(job.spec, "wire", "fp32")
    else:
        plan = job.plan or {}
        model = _plan_model(plan)
        shard, wire = plan.get("shard"), plan.get("wire")
    m4 = m4_unarmed_reason(model)
    if m4 is not None:
        return m4
    if PIECES_ARMED:
        return None
    if shard == "pieces":
        return "shard 'pieces' is not armed: every leg of the pieces tripwire must land first"
    if wire == "native":
        return "wire 'native' is not armed: every leg of the pieces tripwire must land first"
    return None


# A node's transformers RECORD text arrives gzip'd in its fingerprint (about
# 166 KB in base64); its expansion is bounded so a node cannot make the
# coordinator inflate an arbitrary amount.
RECORD_TEXT_MAX_BYTES = 8 << 20
_MISSING = object()


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _canon_sha256(x) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def record_text_of(fp) -> str | None:
    """The transformers RECORD text a fingerprint carries (I-3), or None when
    it carries none or it does not decode within RECORD_TEXT_MAX_BYTES."""
    b64 = _dig(fp, "transformers", "record_gz_b64")
    if not isinstance(b64, str):
        return None
    try:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out = d.decompress(base64.b64decode(b64, validate=True), RECORD_TEXT_MAX_BYTES + 1)
        if len(out) > RECORD_TEXT_MAX_BYTES or d.unconsumed_tail or not d.eof:
            return None
        return out.decode("utf-8")
    except (ValueError, zlib.error):
        return None


def _first_diff(a, b) -> str:
    """The first key (sorted) whose value differs between two dicts, for a
    refusal's text; 'the whole set' when either is not a dict."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return "the whole set"
    for k in sorted(set(a) | set(b), key=str):
        if a.get(k, _MISSING) != b.get(k, _MISSING):
            return str(k)
    return "nothing"


def _node_classes(reps) -> list[str]:
    if not isinstance(reps, list):
        return []
    return sorted(
        {
            r["device_class"]
            for r in reps
            if isinstance(r, dict) and isinstance(r.get("device_class"), str) and r["device_class"]
        }
    )


def hf_admission(
    reports: dict,
    fingerprints: dict,
    builder,
    slot_map: dict,
    pool,
    *,
    torch_records: dict,
    record_diff,
    recorded: dict | None = None,
) -> list[str]:
    """CONTROL-I24 over one admission (I-9). Pure. -> every refusal, each
    naming the node (or the slot) and the cause; [] admits.

    reports: node_id -> its GateReports (one per device it gated);
    fingerprints: node_id -> its hfm.fingerprint (I-3);
    builder: the record builder child's fingerprint, admitted as a node;
    slot_map: slot -> {"node", "device_class"}, plus the slot's recorded
      "torch_records" and "native" after the first launch;
    pool: the node ids of the pool now (the selected nodes and the spares);
    torch_records, record_diff: hfm.TORCH_RECORDS and hfm.record_diff;
    recorded: None at the first launch, then the run's {"hf_code",
      "record_text"}; the references are then the run's, not the builder's.
    """
    out: list[str] = []
    nodes = sorted(set(pool) | {e.get("node") for e in slot_map.values()}, key=str)
    if recorded is not None:
        code = recorded.get("hf_code") or {}
        ref = "the run's recorded hf_code"
        ref_tf = (code.get("transformers"), code.get("transformers_record_sha256"))
        ref_torch = (code.get("torch"), code.get("torch_git_version"))
        ref_text = recorded.get("record_text")
    else:
        ref = "the record builder"
        ref_tf = (
            _dig(builder, "transformers", "version"),
            _dig(builder, "transformers", "record_sha256"),
        )
        ref_torch = (_dig(builder, "torch", "version"), _dig(builder, "torch", "git_version"))
        ref_text = record_text_of(builder)
        if _MISSING in ref_tf + ref_torch:
            return ["the record builder: malformed fingerprint (transformers or torch missing)"]
    ref_probe = _dig(builder, "probe_vector")

    def check_fp(name: str, fp) -> None:
        if not isinstance(fp, dict):
            out.append(f"{name}: no fingerprint")
            return
        tf_v, tf_sha = (
            _dig(fp, "transformers", "version"),
            _dig(fp, "transformers", "record_sha256"),
        )
        t_v, t_git = _dig(fp, "torch", "version"), _dig(fp, "torch", "git_version")
        tag, recs = _dig(fp, "torch", "platform_tag"), _dig(fp, "torch", "records")
        # A peer's fingerprint is untrusted input: the keys are typed before
        # they are hashed or compared (an unhashable tag must not crash).
        if not all(isinstance(x, str) for x in (tf_v, tf_sha, t_v, t_git, tag)) or not (
            isinstance(recs, dict)
        ):
            out.append(f"{name}: malformed fingerprint (transformers or torch missing)")
            return
        # hfm.fingerprint lists every FP-* cause it found (a RECORD file that
        # does not hash, a native object loaded from another path, ...); the
        # builder ran no gate, so this is where its own are refused.
        found = fp.get("refusals")
        if not isinstance(found, list) or found:
            out.append(
                f"{name}: fingerprint refusals {_clip(str(found))}"
                if isinstance(found, list)
                else f"{name}: malformed fingerprint (no refusals list)"
            )
        if tf_v != ref_tf[0]:
            out.append(f"{name}: transformers {tf_v!r} differs from {ref}'s {ref_tf[0]!r}")
        # The texts are compared line by line, not only by sha256: a RECORD
        # whose kept text differs from a node's in one line is refused naming
        # that path even when the sha256 the node reports equals the record's.
        # Fail closed: once there is a reference text (even an emptied one), a
        # node whose text is missing or does not decode is refused, and an
        # empty reference differs from every node's text.
        text = record_text_of(fp)
        if isinstance(ref_text, str) and text is None:
            out.append(f"{name}: transformers RECORD text missing or does not decode")
        path = (
            record_diff(ref_text, text) if isinstance(ref_text, str) and text is not None else None
        )
        if tf_sha != ref_tf[1] or path is not None:
            out.append(
                f"{name}: transformers RECORD differs from {ref}'s at {path}"
                if path
                else f"{name}: transformers RECORD sha256 {str(tf_sha)[:12]} differs from "
                f"{ref}'s {str(ref_tf[1])[:12]}"
            )
        if (t_v, t_git) != ref_torch:
            out.append(
                f"{name}: torch {t_v!r} ({t_git!r}) differs from {ref}'s "
                f"{ref_torch[0]!r} ({ref_torch[1]!r})"
            )
        entry = torch_records.get((tag, t_v))
        if entry is None:
            out.append(
                f"{name}: no TORCH_RECORDS entry for platform {tag!r} torch {t_v!r} "
                "(FP-TORCH-RECORDS)"
            )
        elif recs != entry:
            out.append(
                f"{name}: torch RECORD differs from TORCH_RECORDS[{tag!r}] at "
                f"{_first_diff(recs, entry)}"
            )
        for flag in ("prefix_ok", "code_ok"):
            if fp.get(flag) is False:
                out.append(f"{name}: fingerprint {flag} is false")
        if fp.get("probe_vector") != ref_probe:
            out.append(
                f"{name}: probe vector differs from the record builder's at "
                f"{_first_diff(fp.get('probe_vector'), ref_probe)}"
            )

    check_fp("the record builder", builder)
    groups: dict = {}
    for n in nodes:
        name = f"node {n}"
        reps = reports.get(n)
        if not isinstance(reps, list) or not reps:
            out.append(f"{name}: no gate report")
        else:
            for r in reps:
                if (
                    not isinstance(r, dict)
                    or not isinstance(r.get("device_class"), str)
                    or not r["device_class"]
                ):
                    out.append(f"{name}: malformed gate report")
                elif r.get("ok") is not True:
                    out.append(
                        f"{name}: gate on {r.get('device')!r} failed: "
                        f"{r.get('cause')}: {_clip(str(r.get('detail')))}"
                    )
        fp = fingerprints.get(n)
        check_fp(name, fp)
        tag = _dig(fp, "torch", "platform_tag")
        group = (str(tag), tuple(_node_classes(reps)))
        native = fp.get("native") if isinstance(fp, dict) else None
        if group in groups:
            first, first_native = groups[group]
            if native != first_native:
                out.append(
                    f"{name}: native objects differ from node {first}'s (platform {group[0]}, "
                    f"classes {list(group[1])}) at {_first_diff(native, first_native)}"
                )
        else:
            groups[group] = (n, native)

    need: set = set()
    for slot, e in sorted(slot_map.items(), key=lambda kv: str(kv[0])):
        n, cls = e.get("node"), e.get("device_class")
        if not isinstance(cls, str) or not cls:
            out.append(f"slot {slot}: no recorded device class")
            continue
        need.add(cls)
        held = _node_classes(reports.get(n))
        if cls not in held:
            out.append(
                f"slot {slot}: node {n} does not hold the slot's device class {cls!r} {held}"
            )
        fp = fingerprints.get(n)
        if "torch_records" in e and _dig(fp, "torch", "records") != e["torch_records"]:
            out.append(
                f"slot {slot}: node {n}'s torch RECORD differs from the slot's recorded one at "
                f"{_first_diff(_dig(fp, 'torch', 'records'), e['torch_records'])}"
            )
        if "native" in e and _dig(fp, "native") != e["native"]:
            out.append(
                f"slot {slot}: node {n}'s native objects differ from the slot's recorded ones at "
                f"{_first_diff(_dig(fp, 'native'), e['native'])}"
            )
    if need and not any(need <= set(_node_classes(reports.get(n))) for n in pool):
        out.append(
            f"device map classes {sorted(need)}: no single node of the pool holds them all, so "
            "the map has no oracle (ADR-0009 decision 8)"
        )
    return out


def _hf_pin(model, out: dict) -> None:
    """Pin the builder child's output (I-10) in a spec-mode model: the buffer
    record, the census, hf_code and the mirrored fields. A mirrored field the
    model does not have is refused, never invented."""
    unknown = sorted(k for k in out["fields"] if not hasattr(model, k))
    if unknown:
        raise RuntimeError(f"hf record builder: mirrored fields {unknown} are not model fields")
    model.hf_buffers = out["record"]
    model.hf_census = out["census"]
    model.hf_code = out["hf_code"]
    for k, v in out["fields"].items():
        setattr(model, k, v)


def class_refusals(slot_map: dict, infos: dict) -> list[str]:
    """CONTROL-I25 for arch 'hf': each slot's gate-reported device class
    (slot_map[slot]['device_class']) must equal its node's HELLO device_class
    (infos[node_id]); -> the MIXED-FLEET refusals naming both."""
    out = []
    for slot, e in sorted(slot_map.items()):
        info = infos.get(e.get("node"))
        hello = info.device_class if info is not None else ""
        if e.get("device_class") != hello:
            name = info.name if info is not None else e.get("node")
            out.append(
                f"MIXED-FLEET: slot {slot}: node {name}'s gate reports device class "
                f"{e.get('device_class')!r}, its HELLO {hello!r}"
            )
    return out


def plan_model_mismatch(plan_job: dict, model) -> list[str]:
    """BA-461: every key among arch, hf_config (by canonical sha256), hf_impl
    and moe_* that the plan dict gives (not None, as fleetplan.model_config
    reads it) and the built spec's model (a GPTConfig or a dict) does not
    hold equal, each named. Pure. A key the planner dropped is a refusal,
    never a dense run."""
    keys = [k for k in ("arch", "hf_config", "hf_impl") if plan_job.get(k) is not None]
    keys += sorted(k for k in plan_job if k.startswith("moe_") and plan_job[k] is not None)
    out = []
    for k in keys:
        want = plan_job[k]
        got = model.get(k, _MISSING) if isinstance(model, dict) else getattr(model, k, _MISSING)
        if k == "hf_config":
            a = _canon_sha256(want)
            b = None if got is _MISSING or got is None else _canon_sha256(got)
            if a != b:
                out.append(
                    f"hf_config: the plan's canonical sha256 {a[:12]} != the spec's "
                    f"{b[:12] if b else 'nothing'}"
                )
        elif got is _MISSING:
            out.append(f"{k}: the plan gives {want!r}, the spec's model has no {k}")
        elif _canon(got) != _canon(want):
            out.append(f"{k}: the plan gives {want!r}, the spec's model has {got!r}")
    return out


def replan_rule(
    plan: dict, remaining_steps: int, restart_cost_s: float, replans_done: int, max_replans: int
) -> tuple[bool, str]:
    """CONTROL-I21's rule over a replan() result: apply only when the re-solve
    decided (>= REPLAN_MIN_GAIN shorter, ownership or order changed), the
    predicted gain over the remaining steps exceeds twice the restart cost,
    and the run's rebalance budget is not spent. -> (apply, why)."""
    if replans_done >= max_replans:
        return False, f"the rebalance budget ({max_replans}) is spent"
    if not plan["decision"]:
        return False, str(plan.get("why", "no decision"))
    gain = (plan["pred_cur_s"] - plan["pred_new_s"]) * remaining_steps
    pays = gain > 2 * restart_cost_s
    verdict = "pays" if pays else "does not pay"
    return pays, (
        f"predicted {plan['pred_cur_s']:.6g} s -> {plan['pred_new_s']:.6g} s per step over "
        f"{remaining_steps} step(s): gain {gain:.3g} s {verdict} 2 x restart {restart_cost_s:.3g} s"
    )


def _mem_budgets(spec: JobSpec, groups) -> list:
    """Activation bytes each stage may hold on its nodes (groups[stage] =
    NodeInfos): memory less the reserve and the stage's parameter state. A
    wide stage gets one budget per peer, its own node's, so each lane is
    solved against its own peer and never the stage's sum (BA-29)."""
    from wantrain.scheduler.planner import stage_param_split, vram_reserve

    budgets: list = []
    for r, (a, b) in enumerate(spec.partition):
        trainable, frozen = stage_param_split(spec.model, a, b)
        state = trainable * optim.state_bytes_per_param(spec.optimizer) + frozen * 4
        per = []
        for node in groups[r]:
            mem = node.memory_bytes()
            per.append(max(0, mem - vram_reserve(mem) - state) if mem else 0)
        budgets.append(per if len(per) > 1 else per[0])
    return budgets


# ---------------------------------------------------------------- plan inputs (BA-48)

_HOST_CAPS = ("host_ram_bytes", "host_dram_gbps")
_PCIE_CAPS = ("pcie_h2d_gbps", "pcie_d2h_gbps")
# Every key of a plan-fleet node the rationale lists with its source.
PLAN_INPUT_KEYS = (
    "hostname",
    "site",
    "device",
    "gpu_name",
    "tflops",
    "vram_bytes",
    *_HOST_CAPS,
    *_PCIE_CAPS,
    "nic_up_gbps",
    "nic_down_gbps",
)
_PLANNER_NODE_KEYS = (
    "site",
    "gpu_name",
    *_HOST_CAPS,
    *_PCIE_CAPS,
    "nic_up_gbps",
    "nic_down_gbps",
)
# Pinned host bytes the copy engine holds in flight per peer on CUDA: one chunk
# (runtime/exchange.py copies through pinned memory in CHUNK_BYTES pieces on one
# thread). 0 on cpu and mps, which admission applies itself. HYPOTHESIS, real-
# hardware pending (F7).
PINNED_BYTES_CUDA = float(EXCHANGE_CHUNK_BYTES)


def _reversed_hop(m: dict) -> dict:
    """A hop measurement as seen from its other end: forward and reverse swap."""
    return dict(
        m,
        forward_mbps=m["reverse_mbps"],
        reverse_mbps=m["forward_mbps"],
        bidir_forward_mbps=m["bidir_reverse_mbps"],
        bidir_reverse_mbps=m["bidir_forward_mbps"],
    )


def _rate(x) -> float | None:
    """A worker-reported rate (Mbps) the planner may use: finite and positive."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) and x > 0 else None


def nic_from_hops(infos: list, hops: list) -> dict:
    """node_id -> (up Mbps, down Mbps) from the chain hops' bidirectional
    transfer: hop i joins infos[i] (the dialer, whose up is the forward
    direction) and infos[i + 1]. A node keeps the largest figure per direction
    over its hops; a node no hop measured is left out (its NIC is missing).
    The figures are the dialing worker's report: one that is not a finite
    positive number counts as not measured."""
    nic: dict = {}
    for i, h in enumerate(hops):
        fwd, rev = _rate(h.get("bidir_forward_mbps")), _rate(h.get("bidir_reverse_mbps"))
        if fwd is None or rev is None:
            continue
        for nid, up, down in (
            (infos[i].node_id, fwd, rev),
            (infos[i + 1].node_id, rev, fwd),
        ):
            u, d = nic.get(nid, (0.0, 0.0))
            nic[nid] = (max(u, up), max(d, down))
    return nic


def plan_fleet_nodes(infos: list, nic: dict | None = None) -> list[dict]:
    """The planner's fleet nodes (fleetplan's node keys, BA-48) for the
    registered workers `infos`, rank = position:
      host the dial address, hostname the probed one, site the declared region;
      device and gpu_name as probed (or declared, and marked so);
      host RAM and host DRAM: one figure per hostname, the largest of its
      workers' measurements (each is a lower bound), with that one's source;
      PCIe per direction as probed ('absent' on cpu and mps nodes);
      NIC per direction: `nic` (node_id -> (up, down) Mbps from the link
      probes, 'measured'), or with probing off (nic None) the declared
      uplink and downlink ('declared'); 'missing' when neither exists.
    Every key's source is in `sources`."""
    hosts: dict = {}
    for info in infos:
        cur = hosts.setdefault(info.hostname or info.host, {})
        for fld in _HOST_CAPS:
            val = getattr(info, fld)
            if val is not None and (fld not in cur or val > cur[fld][0]):
                cur[fld] = (val, capacity_source(info, fld))
    out = []
    for i, info in enumerate(infos):
        name = info.hostname or info.host
        sources = {
            "hostname": "measured" if info.hostname else "missing",
            "site": "declared" if info.region else "missing",
        }
        for fld, decl in (
            ("device", "device"),
            ("gpu_name", "gpu_name"),
            ("tflops", "tflops"),
            ("vram_bytes", "vram_bytes"),
        ):
            sources[fld] = "declared" if decl in info.declared else "measured"
        node = dict(
            rank=i,
            host=info.host,
            dial_port=info.data_port,
            listen_port=info.bind_port,
            tflops=info.speed(),
            vram_bytes=info.memory_bytes(),
            compute_cap=list(info.compute_cap) if info.compute_cap else None,
            label=f"{info.name}/{info.gpu_name}",
            device=info.device,
            device_class=info.device_class,
            hostname=name,
            site=info.region or "",
            gpu_name=info.gpu_name,
            declared=list(info.declared),
        )
        for fld in _HOST_CAPS:
            node[fld], sources[fld] = hosts[name].get(fld, (None, "missing"))
        for fld in _PCIE_CAPS:
            node[fld], sources[fld] = getattr(info, fld), capacity_source(info, fld)
        if nic is not None:
            (up, down), src = nic.get(info.node_id, (None, None)), "measured"
        else:
            up, down, src = info.uplink_mbps, info.downlink_mbps, "declared"
        for key, mbps in (("nic_up_gbps", up), ("nic_down_gbps", down)):
            node[key] = None if mbps is None else float(mbps) / 1e3
            sources[key] = "missing" if mbps is None else src
        node["sources"] = sources
        out.append(node)
    return out


def _cap_budgets(budgets: list, groups, cap: dict) -> list:
    """`budgets` (_mem_budgets: per stage a number, or one per peer) with each node's activation budget held to what
    a measured fit calibrated for it (cap: node_id -> bytes, CONTROL-I26)."""
    if not cap:
        return budgets
    out: list = []
    for b, g in zip(budgets, groups):
        nodes = g if isinstance(g, (list, tuple)) else [g]
        per = b if isinstance(b, (list, tuple)) else [b]
        held = [min(x, cap[n.node_id]) if x and n.node_id in cap else x for x, n in zip(per, nodes)]
        out.append(held if isinstance(b, (list, tuple)) else held[0])
    return out


def _stage_fits(spec: JobSpec, groups, peaks: dict | None = None, margin: float = 0.03):
    """fits(s, a, b) for the partition DP (SCHEDULER-I24). A stage whose node has a measured fit peak for its current
    entries is priced from it: the peak less those entries' parameter state (optimizer bytes per trainable element)
    is the node's measured overhead, scaled by the block count; a candidate range fits when its state plus that
    overhead stays within (1 - margin) of the node's fit budget. Otherwise the planner's memory model, at the
    optimizer's bytes per element (16 for adamw priced every adamw-host cut at twice its state)."""
    from wantrain.scheduler.planner import stage_fits, stage_param_split

    bpp = optim.state_bytes_per_param(spec.optimizer)
    n_entries = spec.partition[-1][1]
    state = [bpp * stage_param_split(spec.model, e, e + 1)[0] for e in range(n_entries)]
    blocks = [0 if e in (0, n_entries - 1) else 1 for e in range(n_entries)]
    peaks = peaks or {}

    def measured(n):
        rec = peaks.get(n.node_id)
        if rec is None:
            return None
        (a0, b0), peak, budget = rec
        over = max(0.0, peak - sum(state[a0:b0]))
        return over / max(1, sum(blocks[a0:b0])), budget, (a0, b0), over

    def fits(s, a, b):
        for n in groups[s]:
            m = measured(n)
            if m is not None:
                per_block, budget, (a0, b0), over = m
                extra = per_block * max(0, sum(blocks[a:b]) - sum(blocks[a0:b0]))
                if sum(state[a:b]) + over + extra > (1 - margin) * budget:
                    return False
            elif not stage_fits(
                spec.model,
                a,
                b,
                n.memory_bytes(),
                mb_size=spec.mb_size,
                boundary=spec.model.boundary_dtype,
                state_bpp=bpp,
            ):
                return False
        return True

    return fits


def refined_partition(spec: JobSpec, profiles, fits, min_gain: float = 0.05):
    """The cut proportional to each stage's measured throughput (F + B + W per microbatch, summed over its peers,
    on its own nodes: a profile or the run's own telemetry), under `fits`. -> (new partition, slowest stage before,
    after) in seconds per microbatch, or None when it is the same cut, does not fit, or shortens the slowest stage
    by less than `min_gain`."""
    from wantrain.scheduler.partition import layer_costs, proportional_partition

    costs = layer_costs(spec.model)
    rate = [sum(1.0 / max(1e-9, p["t_f"] + p["t_b"] + p["t_w"]) for p in row) for row in profiles]
    eff = [sum(costs[a:b]) * r for (a, b), r in zip(spec.partition, rate)]
    try:
        new = proportional_partition(spec.model, eff, fits=fits)
    except ValueError:
        return None
    new = [list(x) for x in new]
    before = max(1.0 / r for r in rate)
    after = max(sum(costs[a:b]) / e for (a, b), e in zip(new, eff))
    if new == [list(x) for x in spec.partition] or after > (1.0 - min_gain) * before:
        return None
    return new, before, after


def recut_refusal(spec: JobSpec) -> str | None:
    """Why runtime/recut.py cannot re-cut this run (RUNTIME-I26), or None: width-1 builtin dense chains only."""
    from wantrain.scheduler import autoconfig

    if spec.stages < 2:
        return "one stage"
    if any(w != 1 for w in spec.stage_widths):
        return "a wide stage"
    if spec.shard == "pieces" or spec.model.arch == "hf":
        return "a pieces or hf run"
    if getattr(spec.model, "tie_embeddings", False):
        return "a tied embedding"
    if autoconfig.moe_fields(spec.model) is not None:
        return "an MoE model"
    return None


def stage_compute_ms(summary: dict) -> float:
    """A slot's compute per microbatch in one WT_TELEM summary: its F + B + W medians, in ms."""
    ops = summary.get("ops") or {}
    return float(sum((ops.get(k) or {}).get("med_ms", 0.0) for k in ("F", "B", "W")))


def slow_slots(hist: dict, base: dict, ratio: float = 1.3, runs: int = 3) -> list:
    """CONTROL-I27: the slots whose last `runs` compute samples all exceed `ratio` x their baseline. hist: slot ->
    [ms, ...]; base: slot -> ms. -> [(slot, latest / baseline)]. A noisy host slows its stage, and through the
    pipeline every other one (2026-10-12: one stage's backward swung 128-1,719 ms while its twin held 122)."""
    out = []
    for slot, xs in hist.items():
        b = base.get(slot)
        tail = xs[len(xs) - runs :] if len(xs) >= runs else []
        if b and tail and all(x > ratio * b for x in tail):
            out.append((slot, round(xs[-1] / b, 3)))
    return out


def scheduled_pending_peak(spec: JobSpec, rank: int, peer: int) -> float:
    """The zero-bubble memory model's peak of slot (rank, peer) over its pinned order: the bytes its in-flight
    microbatches hold at worst, as the solve priced them (stage_memory_model replayed over schedule_order)."""
    from wantrain.scheduler.zero_bubble import stage_memory_model

    a, b = spec.partition[rank]
    mf, mb, mw = stage_memory_model(spec, b - a, (a, b), spec.recomputes(rank))
    delta = {"F": mf, "B": mb, "W": mw}
    m = peak = 0.0
    for op, _mb in spec.schedule_order[rank][peer]:
        m += delta[op]
        peak = max(peak, m)
    return peak


def calibrated_budget(
    spec: JobSpec, rank: int, peer: int, peak: float, budget: float, oom: bool
) -> int:
    """CONTROL-I26: the activation budget, in the schedule's own memory model, that keeps slot (rank, peer) under its
    fit budget. The fit measured `peak` (for an out-of-memory fit, the whole card: a lower bound) under the pinned
    order; past the stage's state (weights, folded gradient, optimizer) that is what its in-flight microbatches and
    their working set really took, `factor` times what the model priced. The budget is the room under the fit budget
    divided by that factor (an out-of-memory fit, whose demand is unknown, another MEM_FEEDBACK), so the model's
    pending microbatches, at their measured size, fit."""
    from wantrain.scheduler.planner import stage_param_split

    a, b = spec.partition[rank]
    trainable, frozen = stage_param_split(spec.model, a, b)
    state = trainable * optim.state_bytes_per_param(spec.optimizer) + frozen * 4
    modeled = max(scheduled_pending_peak(spec, rank, peer), 1.0)
    factor = max(1.0, (peak - state) / modeled) * (MEM_FEEDBACK if oom else 1.0)
    return int(max(0.0, budget - state) / factor)


def shrink_for_overshoot(nodes: list[dict], node_ids: list, over: dict) -> list[str]:
    """Feed measured fit overshoots back into the planner's fleet: each overflowing node's vram_bytes (what the
    planner's cut sees, `nodes[i]` for `node_ids[i]`) drops by MEM_FEEDBACK x (peak - budget), so the next cut moves
    layers off it. The fit budget stays the node's real memory (NodeInfo.memory_bytes). -> one note per node."""
    notes = []
    for node, nid in zip(nodes, node_ids):
        if nid not in over or not node.get("vram_bytes"):
            continue
        peak, budget = over[nid]
        cut = int(MEM_FEEDBACK * (peak - budget))
        node["vram_bytes"] = max(0, int(node["vram_bytes"]) - cut)
        notes.append(
            f"{node.get('label') or nid} plans with {node['vram_bytes'] / 2**30:.1f} GiB (-{cut / 2**30:.2f})"
        )
    return notes


def missing_measurements(nodes: list[dict]) -> list[str]:
    """Every plan input that is neither measured nor absent, as the I16
    refusal names it ('<source>: host <hostname> dram', '<source>: <label>
    nic_up'): what a pieces candidate is refused on (BA-48)."""
    out: list[str] = []
    for n in nodes:
        for fld, key in (("host_ram_bytes", "ram_bytes"), ("host_dram_gbps", "dram")):
            src = n["sources"][fld]
            text = f"{src}: host {n['hostname']} {key}"
            if src not in admission.PIECES_SOURCES and text not in out:
                out.append(text)
        for fld in (*_PCIE_CAPS, "nic_up_gbps", "nic_down_gbps"):
            src = n["sources"][fld]
            if src not in admission.PIECES_SOURCES:
                out.append(f"{src}: {n['label']} {fld.removesuffix('_gbps')}")
    return out


def _nodes_by_id(infos: list, nic: dict | None) -> dict:
    return {n.node_id: d for n, d in zip(infos, plan_fleet_nodes(infos, nic))}


def plan_inputs(nodes: list[dict], fleet: dict, link_source: str) -> dict:
    """Every input of a plan with its source, for the rationale."""
    return {
        "nodes": [
            {
                "node": n["rank"],
                "label": n["label"],
                **{k: {"value": n.get(k), "source": n["sources"][k]} for k in PLAN_INPUT_KEYS},
            }
            for n in nodes
        ],
        "k_passes": dict(fleet["k_passes"]),
        "link": dict(fleet["link"], source=link_source),
    }


# ---------------------------------------------------------------- pieces re-checks (I16)


def _planner_fleet(nodes: list[dict], adapt_budget_ms: float = 0.0):
    """Plan-fleet node dicts -> the planner's Fleet, read as fleetplan reads
    them; k is the transport's counted figure (SC-5)."""
    from wantrain.scheduler import planner

    return planner.Fleet(
        nodes=[
            planner.Node(
                host=n["host"],
                tflops=float(n["tflops"]),
                vram_bytes=int(n["vram_bytes"]),
                dial_port=int(n.get("dial_port", 0)),
                listen_port=int(n.get("listen_port", 0)),
                label=n["label"],
                compute_cap=tuple(n["compute_cap"]) if n.get("compute_cap") else None,
                device=n["device"],
                device_class=n.get("device_class") or "",
                hostname=n["hostname"],
                declared=tuple(n.get("declared") or ()),
                sources=dict(n["sources"]),
                **{k: n[k] for k in _PLANNER_NODE_KEYS if n.get(k) is not None},
            )
            for n in nodes
        ],
        link_mbps=1e6,
        rtt_ms=0.0,
        duplex="per_direction",
        k_passes=(float(admission.counted_k(adapt_budget_ms)), "counted"),
    )


def pieces_admission(
    spec: JobSpec, groups: list, nodes_by_id: dict, s2_refuses: bool
) -> tuple[list, list]:
    """SCHEDULER-I16 over the wide stages of a placed shard 'pieces' spec, as
    it will run: its mb_owner, peer weights and slot orders (BA-44, BA-47,
    BA-50). groups[stage][peer] are the slots' NodeInfos, nodes_by_id their
    plan-fleet nodes (plan_fleet_nodes).
    Refused: a stage whose peers are not one probed class (check_class,
    probed: a declared device or gpu_name never passes), device memory and
    MEM-HOST (pinned copy chunks on CUDA) at each peer's M_k, and a host whose
    RAM is not known. S2 from the exchange simulation of these orders, and the
    S1 links, are refusals when `s2_refuses` (plan mode, after the profiled
    re-weighting) and warnings otherwise (spec mode); a missing measurement
    S1 and S2 need is then reported in their place. -> (refusals, warnings),
    admission.Reason lists. Nothing for any other spec."""
    import dataclasses
    import types

    from wantrain.scheduler import planner
    from wantrain.scheduler.exchange_sim import Fabric
    from wantrain.scheduler.zero_bubble import ZBConfig, stage_memory_model

    widths = [len(g) for g in groups]
    if spec.shard != "pieces" or max(widths) < 2:
        return [], []
    flat = [n for g in groups for n in g]
    fleet = _planner_fleet([nodes_by_id[n.node_id] for n in flat], spec.adapt_budget_ms)
    ctx = planner._Ctx(
        spec.model, fleet, {"spec": spec, "rationale": {"k_inflight": 1}}, spec.device
    )
    rows, k = [], 0
    for w in widths:
        rows.append(list(range(k, k + w)))
        k += w
    S, M = spec.stages, spec.microbatches
    stages = [
        ctx.stage(
            r,
            S,
            tuple(spec.partition[r]),
            rows[r],
            spec.shard if widths[r] > 1 else "zero1",
            spec.wire,
        )
        for r in range(S)
    ]
    wide = [st for st in stages if st.width > 1]
    refusals: list = []
    for st in wide:
        for r in admission.check_class(st.peers, probed=True):
            refusals.append(
                dataclasses.replace(r, where=f"stage {st.index}: {r.where}", stage=st.index)
            )
    owned = admission.mb_owned_from(spec.mb_owner, stages)
    refusals += admission.device_memory(wide, owned, S, M)
    refusals += admission.mem_host(wide, ctx.hosts, owned, M, PINNED_BYTES_CUDA)
    missing = admission.missing_inputs(wide, ctx.hosts, ctx.k_source, True)
    refusals += [r for r in missing if r.where.endswith("ram_bytes")]
    net_missing = [r for r in missing if not r.where.endswith("ram_bytes")]
    timing: list = list(net_missing)
    if not net_missing:
        # C and the orders' timing from the planner's cost model, in seconds
        # at each stage's aggregate rate, as the planner admitted the plan.
        costs = [ctx.costs(a, b) for a, b in spec.partition]
        agg = [sum(g.tflops for g in st.peers) * 1e12 for st in stages]
        mems = [
            stage_memory_model(
                types.SimpleNamespace(
                    model=spec.model, mb_size=spec.mb_size, recompute=spec.recompute
                ),
                b - a,
            )
            for a, b in spec.partition
        ]
        comm, _floor = planner._hops(ctx, {"S": S, "w": max(widths)}, stages)
        zcfg = ZBConfig(
            S,
            M,
            [c[0] / g for c, g in zip(costs, agg)],
            [c[1] / g for c, g in zip(costs, agg)],
            [c[2] / g for c, g in zip(costs, agg)],
            comm=comm,
            mem_f=[x[0] for x in mems],
            mem_b=[x[1] for x in mems],
            mem_w=[x[2] for x in mems],
        )
        fabric = Fabric(
            host_dram={h: (x.dram or math.inf) for h, x in ctx.hosts.items()},
            uplinks=planner._uplinks(ctx),
            k_passes=ctx.k,
            rtt=ctx.rtt,
            site_rtt={(a, b): ctx.site_rtt for a in ctx.sites for b in ctx.sites if a != b},
        )
        orders = spec.emission_order or ["store"] * S  # finalize fills it
        by_spec = {v: k for k, v in planner.EMISSION_SPEC.items()}
        # The simulation takes one emission order for every stage; a spec may
        # give its wide stages different ones, so each stage's S2 is read from
        # the run under its own order.
        for order in sorted({orders[st.index] for st in wide}):
            mine = {st.index for st in wide if orders[st.index] == order}
            timing += [
                r
                for r in admission.recheck(
                    wide,
                    ctx.hosts,
                    zcfg,
                    spec.schedule_order,
                    spec.peer_weights,
                    spec.mb_owner,
                    fabric,
                    emission=by_spec[order],
                    pinned_bytes_per_peer=PINNED_BYTES_CUDA,
                )
                if r.kind == "S2" and r.stage in mine
            ]
        if not s2_refuses:
            timing += admission.s1(wide, ctx.hosts, ctx.k, planner._uplinks(ctx))
    if s2_refuses:
        return refusals + timing, []
    return refusals, timing


def _is_int(x) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


def _telem_number(x, hi, integer=False) -> bool:
    """A real number in [0, hi]; NaN and infinities fail the comparison."""
    if isinstance(x, bool) or not isinstance(x, int if integer else (int, float)):
        return False
    return 0 <= x <= hi


def _telem_hop_label(label) -> bool:
    """next<q>, prev<q> or intra<k>, as the stage runtime names its connections."""
    if not isinstance(label, str) or len(label) > 16:
        return False
    head = label.rstrip("0123456789")
    return head in ("next", "prev", "intra") and len(head) < len(label)


def _telem_summary_error(summary, slot, start_step: int, final_step) -> str | None:
    """Why a TELEM_STEP summary is refused, or None (CONTROL-I18). The slot is
    the connection's; the summary must name it, not the reverse."""
    if not isinstance(summary, dict):
        return "summary is not an object"
    if len(json.dumps(summary, separators=(",", ":"))) > TELEM_MAX_BYTES:
        return f"summary above {TELEM_MAX_BYTES} bytes"
    if set(summary) != TELEM_KEYS:
        return f"summary keys {sorted(summary)}"
    v, step = summary["v"], summary["step"]
    if isinstance(v, bool) or v != 1 or not isinstance(v, int):
        return f"version {v!r}"
    for k, want in (("rank", slot[0]), ("peer", slot[1])):
        got = summary[k]
        if isinstance(got, bool) or not isinstance(got, int) or got != want:
            return f"{k} {got!r} is not this connection's slot {slot}"
    if isinstance(step, bool) or not isinstance(step, int):
        return f"step {step!r}"
    if step < start_step or (final_step is not None and step > final_step):
        return f"step {step} outside [{start_step}, {final_step}]"
    ms = [summary["step_ms"], summary["stall_ms"]]
    counts = []
    if not _telem_number(summary["emit_us"], TELEM_MAX_MS * 1e3):
        return "emit_us outside [0, 1e9] us or not finite"
    ops = summary["ops"]
    if not isinstance(ops, dict) or set(ops) != TELEM_OPS:
        return "ops keys"
    for op in ops.values():
        if not isinstance(op, dict) or set(op) != TELEM_OP_KEYS:
            return "op keys"
        counts.append(op["n"])
        ms.append(op["med_ms"])
    hops = summary["hops"]
    if not isinstance(hops, dict):
        return "hops is not an object"
    for label, h in hops.items():
        if not _telem_hop_label(label):
            return f"hop label {label!r}"
        if not isinstance(h, dict) or set(h) != TELEM_HOP_KEYS:
            return "hop keys"
        counts += [h["frames"], h["bytes"]]
        ms.append(h["busy_ms"])
    if not all(_telem_number(x, TELEM_MAX_MS) for x in ms):
        return "a time outside [0, 1e6] ms or not finite"
    if not all(_telem_number(x, TELEM_MAX_COUNT, integer=True) for x in counts):
        return "a count outside [0, 1e12] or not an integer"
    return None


def _catch(errs, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        errs.append(e)
