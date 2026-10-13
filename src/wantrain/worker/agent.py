"""Worker agent: the node-side daemon of the control plane.

Lifecycle: start -> measure the node (resources/probe.py) -> dial the
coordinator and register (HELLO with capabilities) -> heartbeat -> answer link
probes -> on ASSIGN run its stage as a supervised child process -> stream
progress, ship every checkpoint wave, report the exit -> go idle for the next
assignment. The training numerics never pass through the agent: it is a
supervisor, so it can kill and relaunch a stage (abort, heal, restart in
place) without itself dying, and a stage keeps training through a control-
plane blip while the agent reconnects.

No code or configuration lives on the worker beyond this package: everything
a stage does is a function of the spec it is handed.
"""

from __future__ import annotations

import functools
import ipaddress
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from wantrain.fault_tolerance import checkpoint
from wantrain.resources import links
from wantrain.transport.control import (
    Channel,
    ChannelClosed,
    control_dial,
    session_mac,
    token_mac,
)
from wantrain.transport.secure import hardware_aes

HEARTBEAT_S = 1.0
LOG_TAIL_BYTES = 4000
# A WT_TELEM line above this many bytes is dropped, never forwarded (CONTROL-I18).
TELEM_LINE_MAX = 4096
SIM_EXIT_DELAY_ENV = "WANTRAIN_SIM_EXIT_DELAY_S"
# Backstop scan for unshipped waves; each WT_WAVE line wakes the shipper at once.
SHIP_SCAN_S = 0.5
# Bound on CHALLENGE, HELLO and WELCOME: nothing before admission is large.
HANDSHAKE_MSG_MAX = 64 << 10
_PEER_ID_RE = re.compile(r"[0-9a-f]{32}")
# GATE (I-9, BA-428): request and reply bounds, the gate child's timeout.
GATE_CONFIG_MAX = 256 << 10
GATE_REPLY_MAX = 1 << 20
GATE_TIMEOUT_S = 300.0
GATE_RECIPE_KEYS = frozenset({"compute_dtype", "seq_len", "mb_size"})
# A run id names a directory under the workdir: no separator, no dot-dot.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_gate_request(req, own_devices) -> list[str]:
    """Why a GATE request (I-9) is malformed; [] when it is well formed."""
    if not isinstance(req, dict):
        return [f"request is a {type(req).__name__}, not an object"]
    errs = []
    if req.get("kind") != "GATE":
        errs.append(f"kind {str(req.get('kind'))[:40]!r} is not 'GATE'")
    run_id = req.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        errs.append(f"run_id {str(run_id)[:40]!r} is not 1-128 of [A-Za-z0-9._-]")
    hf = req.get("hf_config")
    if not isinstance(hf, dict):
        errs.append(f"hf_config is a {type(hf).__name__}, not an object")
    else:
        n = len(json.dumps(hf).encode())
        if n > GATE_CONFIG_MAX:
            errs.append(f"hf_config is {n} bytes, above {GATE_CONFIG_MAX}")
    recipe = req.get("recipe")
    if not isinstance(recipe, dict):
        errs.append(f"recipe is a {type(recipe).__name__}, not an object")
    else:
        if set(recipe) != GATE_RECIPE_KEYS:
            errs.append(
                f"recipe keys {sorted(map(str, recipe))} are not {sorted(GATE_RECIPE_KEYS)}"
            )
        cd = recipe.get("compute_dtype")
        if "compute_dtype" in recipe and not (isinstance(cd, str) and 0 < len(cd) <= 32):
            errs.append(f"recipe compute_dtype {str(cd)[:40]!r} is not a dtype name")
        for k in ("seq_len", "mb_size"):
            v = recipe.get(k)
            if k in recipe and not (type(v) is int and v > 0):
                errs.append(f"recipe {k} {str(v)[:40]!r} is not a positive integer")
    devices = req.get("devices")
    if not isinstance(devices, list) or not devices:
        errs.append("devices is not a non-empty list")
    else:
        own = set(own_devices)
        for d in devices:
            if not isinstance(d, str) or d not in own:
                errs.append(f"device {str(d)[:40]!r} is not one of this node's {sorted(own)}")
        if len(set(map(str, devices))) != len(devices):
            errs.append("devices repeats a device")
    return errs


def read_gate_report(path: str, device: str) -> dict:
    """The gate child's report for `device`, read only when within the reply
    bound. The child (`hfm gate`, one device) writes {"reports": [GateReport],
    ...}; that one report is returned. ValueError names what is wrong with it."""
    try:
        n = os.path.getsize(path)
    except OSError:
        raise ValueError(f"gate child on {device} wrote no report") from None
    if n > GATE_REPLY_MAX:
        raise ValueError(f"gate report on {device} is {n} bytes, above {GATE_REPLY_MAX}")
    with open(path, "rb") as f:
        try:
            out = json.loads(f.read())
        except ValueError as e:
            raise ValueError(f"gate report on {device} is not JSON: {e}") from None
    reps = out.get("reports") if isinstance(out, dict) else None
    if not isinstance(reps, list) or len(reps) != 1:
        raise ValueError(f"gate output on {device} does not hold exactly one report")
    rep = reps[0]
    if not isinstance(rep, dict) or not isinstance(rep.get("device_class"), str):
        raise ValueError(f"gate report on {device} is not a GateReport with a device_class")
    if rep.get("device") != device:
        raise ValueError(f"gate report for {str(rep.get('device'))[:40]!r}, asked for {device!r}")
    return rep


def gate_reply(rid, run_id: str, reports: list[dict]) -> dict:
    """GATE_REPORT (I-9): the reports in request order, the node's fingerprint
    (its CPU report's, else the first) and the device classes. ValueError when
    the encoded reply exceeds GATE_REPLY_MAX."""
    fp_from = next((r for r in reports if r.get("device") == "cpu"), reports[0])
    reply = {
        "kind": "GATE_REPORT",
        "rid": rid,
        "run_id": run_id,
        "reports": reports,
        "fingerprint": fp_from.get("fingerprint"),
        "device_classes": [r["device_class"] for r in reports],
    }
    n = len(json.dumps(reply).encode())
    if n > GATE_REPLY_MAX:
        raise ValueError(f"GATE_REPORT is {n} bytes, above {GATE_REPLY_MAX}")
    return reply


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class ChildAborted(RuntimeError):
    pass


class ChildTracker:
    """The running children an ABORT kills. abort() closes the current epoch,
    so a child started for an aborted epoch is killed at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: dict = {}
        self.epoch = 0

    def add(self, proc: subprocess.Popen, prefix: str, epoch: int) -> bool:
        with self._lock:
            if epoch != self.epoch:
                return False
            self._procs[proc] = prefix
            return True

    def discard(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.pop(proc, None)

    def abort(self) -> None:
        """Kill every tracked child's process group and remove its prefix."""
        with self._lock:
            self.epoch += 1
            procs, self._procs = self._procs, {}
        for proc, prefix in procs.items():
            _kill_group(proc)
            proc.wait()
            shutil.rmtree(prefix, ignore_errors=True)


def run_child(args, env, prefix_root: str, timeout: float, tracker=None, epoch: int = 0):
    """Run one child under a fresh pycache prefix in prefix_root (BA-423):
    `args(prefix)` is its argv. The prefix is removed when the child exits,
    on failure and on timeout too; a timeout kills the child's process group
    and raises subprocess.TimeoutExpired. -> subprocess.CompletedProcess."""
    os.makedirs(prefix_root, exist_ok=True)
    prefix = tempfile.mkdtemp(prefix="pyc-", dir=prefix_root)
    try:
        argv = list(args(prefix))
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            if tracker is not None and not tracker.add(proc, prefix, epoch):
                _kill_group(proc)
                proc.communicate()
                raise ChildAborted("aborted before the child ran")
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                raise
        finally:
            if tracker is not None:
                tracker.discard(proc)
            if proc.poll() is None:
                _kill_group(proc)
                proc.wait()
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)
    finally:
        shutil.rmtree(prefix, ignore_errors=True)


def _is_hf(spec) -> bool:
    """Whether a spec dict's model is arch "hf"."""
    model = spec.get("model") if isinstance(spec, dict) else None
    return isinstance(model, dict) and model.get("arch") == "hf"


def is_loopback(host: str) -> bool:
    """Local mode (CONTROL-I16): 127.0.0.0/8, ::1 or the name localhost. No
    other name is resolved, so a name that points at loopback is not local."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def sim_exit_delay_s(environ=os.environ) -> float:
    """Test seam (timing only): seconds a cleanly exited stage's EXIT is held
    back after its last checkpoint ship, as after a slow final-wave transfer."""
    raw = environ.get(SIM_EXIT_DELAY_ENV)
    if raw is None:
        return 0.0
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{SIM_EXIT_DELAY_ENV}={raw!r} is not a number") from None
    if not 0.0 <= v <= 3600.0:
        raise ValueError(f"{SIM_EXIT_DELAY_ENV}={raw!r} is outside [0, 3600]")
    return v


class StageProcess:
    """One supervised stage child: stdout captured to a log, markers parsed."""

    def __init__(self, cmd, log_path, env, on_line, on_exit=None, prefix=None):
        self.log_path = log_path
        # An hf stage's pycache prefix, removed once the child has exited.
        self.prefix = prefix
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log = open(log_path, "ab")
        self.done_marker = False
        self.rc = None
        self.reported = False
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except BaseException:
            self._log.close()
            self._remove_prefix()
            raise
        self._on_line = on_line
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        # Set (under the agent's lock) by whoever sends this stage's EXIT.
        self.exit_claimed = False
        if on_exit is not None:
            # Reaps the child the moment it ends (a kill included) and calls
            # on_exit(self) once its output is drained, so the exit is reported
            # at once, not behind a scan or a wave on the wire.
            threading.Thread(target=self._reap, args=(on_exit,), daemon=True).start()

    def _reap(self, on_exit):
        self.wait()
        on_exit(self)

    def _pump(self):
        assert self.proc.stdout is not None
        with self.proc.stdout:  # the pipe closes at EOF, a pump error included
            for raw in self.proc.stdout:
                self._write_log(raw)
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("WT_DONE"):
                    self.done_marker = True
                try:
                    self._on_line(line)
                except Exception:  # noqa: BLE001 -- a reporting hiccup must not kill the pump
                    pass

    def _write_log(self, raw: bytes) -> None:
        """Append one line to the stage log. A failed write (a full disk) stops the log, once and said, never the
        pump: the pump also forwards the stage's telemetry, progress and wave announcements, and a dead pump
        closes the stage's stdout (2026-10-13: a full disk froze one stage's log, and its reports, at step 21)."""
        log = self._log
        if log is None:
            return
        try:
            log.write(raw)
            log.flush()
        except (OSError, ValueError) as e:  # ValueError: closed under us by close()
            self._log = None
            if not log.closed:
                print(
                    f"[agent] stage log {self.log_path} stopped: {e}", file=sys.stderr, flush=True
                )
            try:
                log.close()
            except OSError:
                pass

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self) -> int:
        rc = self.proc.wait()
        self._reader.join(timeout=10)
        self.rc = rc
        self._remove_prefix()
        return rc

    def _remove_prefix(self) -> None:
        if self.prefix is not None:
            shutil.rmtree(self.prefix, ignore_errors=True)

    def kill(self, grace_s: float = 5.0) -> None:
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait()

    def tail(self) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                n = f.tell()
                f.seek(max(0, n - LOG_TAIL_BYTES))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def close(self):
        log, self._log = self._log, None
        try:
            if log is not None:
                log.close()
        except OSError:
            pass


def probe_bind(advertised: int, listen: int, serve) -> int:
    """The probe port a worker advertises, after starting its echo server with `serve(port) -> bound port`.

    advertised < 0: no probe server. listen > 0 binds there and advertises `advertised` (a NAT or container port map:
    the data plane's --listen-port / --data-port split, for the probe); otherwise the server binds `advertised` (0 = any
    free port) and the bound port is what is advertised."""
    if advertised < 0:
        return advertised
    if listen > 0:
        serve(listen)
        return advertised
    return serve(advertised)


class WorkerAgent:
    def __init__(
        self,
        coordinator: str,
        advertise_host: str,
        data_port: int,
        listen_port: int = 0,
        probe_port: int = 0,
        name: str = "",
        workdir: str = "wantrain-work",
        probe_listen_port: int = 0,
        token: str | None = None,
        device: str | None = None,
        declare: dict | None = None,
        emulation: dict | None = None,
        bench: bool = True,
        reconnect_window_s: float = 300.0,
        python: str | None = None,
        coordinator_id: str | None = None,
    ):
        host, port = coordinator.rsplit(":", 1)
        self.coord = (host, int(port))
        # The coordinator's pinned key id (CONTROL-I16): required off loopback;
        # on loopback without one, any coordinator key is accepted (local mode).
        cid = coordinator_id if coordinator_id is not None else os.environ.get("WANTRAIN_COORD_ID")
        cid = cid or None
        if cid is not None and not _PEER_ID_RE.fullmatch(cid):
            raise ValueError(f"coordinator id {cid[:40]!r} is not 32 lowercase hex digits")
        if cid is None and not is_loopback(host):
            raise ValueError(
                f"coordinator {host} is not on loopback: pin its key with --coordinator-id "
                "(or WANTRAIN_COORD_ID), the id it prints as COORDINATOR_ID"
            )
        self.coordinator_id = cid
        self.advertise_host = advertise_host
        self.data_port = data_port
        self.listen_port = listen_port
        self.probe_port = probe_port
        self.probe_listen_port = probe_listen_port
        self.name = name or f"{platform.node()}-{data_port}"
        self.workdir = os.path.abspath(workdir)
        # This agent's own identity, for its control handshake.
        self.key_path = os.path.join(self.workdir, "peer_key")
        self.token = token if token is not None else os.environ.get("WANTRAIN_COORD_TOKEN")
        self.device = device
        self.declare = dict(declare or {})
        emu = emulation or {}
        self.emu = links.Emulation(
            emu.get("region") or self.declare.get("region"), emu.get("profile")
        )
        if self.emu.region and "region" not in self.declare:
            self.declare["region"] = self.emu.region
        self.bench = bench
        self.reconnect_window_s = reconnect_window_s
        self.python = python or sys.executable
        self.sim_exit_delay_s = sim_exit_delay_s()

        self.node_id = None
        self.session = None
        self.ch: Channel | None = None
        self._stop = threading.Event()
        self._probe_stop = threading.Event()
        self._lock = threading.RLock()
        self.stage: StageProcess | None = None
        self.assignment: dict | None = None
        self.last_step: int | None = None
        # (run_id, rank, peer, step) of every wave shipped (or restored, so
        # held by the coordinator already), of every wave announced or found
        # on disk, and of every wave given up on (a ship gap, logged once).
        self._shipped: set = set()
        self._seen: set = set()
        self._given_up: set = set()
        # One shipper at a time (supervisor, ABORT) and no ship across a
        # RESTORE: _shipped and the slot's waves have a single user (CONTROL-I23).
        self._ship_lock = threading.Lock()
        self._ship_wake = threading.Event()
        # (RESTORE_PART header, received body path) held until the RESTORE.
        self._restore_parts: list = []
        self.probe_info: dict = {}
        # Running GATE children: ABORT and shutdown kill them (I-9).
        self._gate_children = ChildTracker()

    # ------------------------------------------------------------ connection
    def _hello(self, transcript: bytes) -> dict:
        h = {
            "kind": "HELLO",
            "name": self.name,
            "advertise_host": self.advertise_host,
            "data_port": self.data_port,
            "listen_port": self.listen_port,
            "probe_port": self.probe_port,
            "probe": self.probe_info,
            "declare": self.declare,
            "pid": os.getpid(),
        }
        if self.emu.enabled:
            h["emulation"] = self.emu.to_dict()
        if self.node_id and self.session:
            # The session is proven over this transcript, never sent.
            h["node_id"] = self.node_id
            h["session_mac"] = session_mac(self.session, transcript)
            with self._lock:
                if self.assignment:
                    h["assignment"] = {
                        k: self.assignment[k] for k in ("run_id", "attempt", "rank", "peer")
                    }
                    h["stage_alive"] = bool(self.stage and self.stage.alive())
        return h

    def _connect(self) -> Channel:
        """Control handshake against the pinned coordinator key, so the channel
        is sealed before any message; then CHALLENGE, HELLO (token MAC and, on
        a reconnect, session MAC over the transcript) and WELCOME (CONTROL-I16)."""
        sock = socket.create_connection(self.coord, timeout=15)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            ch, info = control_dial(sock, self.key_path, self.coordinator_id, hardware_aes())
            msg, _ = ch.recv(max_inline=HANDSHAKE_MSG_MAX)
            if msg["kind"] != "CHALLENGE":
                raise ChannelClosed(f"expected CHALLENGE, got {msg['kind']}")
            auth = msg.get("auth")
            if self.token and auth is not True:
                raise ChannelClosed("coordinator asked for no token (downgrade): refused")
            transcript = info["transcript"]
            hello = self._hello(transcript)
            if auth:
                if not self.token:
                    raise ChannelClosed("coordinator requires a token (WANTRAIN_COORD_TOKEN)")
                hello["mac"] = token_mac(self.token, transcript)
            ch.send(hello)
            msg, _ = ch.recv(max_inline=HANDSHAKE_MSG_MAX)
            if msg["kind"] == "REJECT":
                raise PermissionError(f"coordinator rejected this worker: {msg.get('reason')}")
            if msg["kind"] != "WELCOME":
                raise ChannelClosed(f"expected WELCOME, got {msg['kind']}")
            sock.settimeout(None)
        except BaseException:
            sock.close()
            raise
        self.node_id, self.session = msg["node_id"], msg["session"]
        return ch

    def send(self, msg: dict, body: bytes = b"") -> bool:
        ch = self.ch
        if ch is None:
            return False
        try:
            ch.send(msg, body)
            return True
        except ChannelClosed:
            return False

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        # probe_port 0 = pick a free port, negative = no probe server.
        self.probe_port = probe_bind(
            self.probe_port, self.probe_listen_port, lambda p: links.serve_echo(p, self._probe_stop)
        )
        from wantrain.resources.probe import measure

        self.probe_info = measure(self.device, bench=self.bench)
        if not self.probe_info.get("ok"):
            raise RuntimeError(f"node probe failed: {self.probe_info.get('error')}")
        threading.Thread(target=self._heartbeat, daemon=True).start()
        deadline = None
        while not self._stop.is_set():
            try:
                self.ch = self._connect()
                deadline = None
                print(f"[agent {self.name}] registered as {self.node_id}", flush=True)
                self._serve(self.ch)
            except PermissionError:
                raise
            except (OSError, ChannelClosed, ValueError) as e:
                ch, self.ch = self.ch, None
                if ch is not None:  # the lost channel's socket closes now, not at GC
                    ch.close()
                if self._stop.is_set():
                    break
                print(f"[agent {self.name}] coordinator connection: {e}", flush=True)
                deadline = deadline or time.monotonic() + self.reconnect_window_s
                if time.monotonic() > deadline:
                    print(
                        f"[agent {self.name}] coordinator unreachable for "
                        f"{self.reconnect_window_s:.0f}s; giving up ({e})",
                        flush=True,
                    )
                    break
                time.sleep(1.0)
        self._shutdown()

    def _serve(self, ch: Channel) -> None:
        while not self._stop.is_set():
            msg, body = ch.recv(file_dir=os.path.join(self.workdir, "incoming"))
            kind = msg["kind"]
            handler = getattr(self, f"_on_{kind.lower()}", None)
            if handler is None:
                self.send(
                    {"kind": "ERROR", "rid": msg.get("rid"), "error": f"unknown message {kind}"}
                )
                continue
            if kind == "GATE":
                # The epoch as received: an ABORT read after this GATE kills it.
                handler = functools.partial(handler, epoch=self._gate_children.epoch)
            if kind in ("PROBE", "MEASURE", "FIT", "PROFILE", "GATE"):
                # Slow requests run off the reader so heartbeats and aborts flow.
                threading.Thread(target=self._guard, args=(handler, msg, body), daemon=True).start()
            else:
                self._guard(handler, msg, body)

    def _guard(self, handler, msg, body):
        try:
            handler(msg, body)
        except Exception as e:  # noqa: BLE001 -- reported to the coordinator
            self.send({"kind": "ERROR", "rid": msg.get("rid"), "error": f"{type(e).__name__}: {e}"})

    def _heartbeat(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                a = self.assignment
                # Alive until its exit has been reported: the EXIT message is
                # the authority on how a stage ended, not the heartbeat.
                stage = self.stage
                hb = {
                    "kind": "HEARTBEAT",
                    "t": time.time(),
                    "stage_alive": bool(stage and (stage.alive() or not stage.reported)),
                    "last_step": self.last_step,
                }
                if a:
                    hb.update(run_id=a["run_id"], attempt=a["attempt"])
            self.send(hb)
            self._stop.wait(HEARTBEAT_S)

    def _shutdown(self) -> None:
        self._probe_stop.set()
        self._gate_children.abort()
        with self._lock:
            if self.stage:
                self.stage.kill()

    # ------------------------------------------------------------ requests
    def _on_ping(self, msg, body):
        self.send({"kind": "PONG", "rid": msg.get("rid")})

    def _on_probe(self, msg, body):
        targets = msg["targets"]
        row = links.measure_row(
            targets,
            msg["self_index"],
            self.emu,
            int(msg.get("bulk_bytes", links.PROBE_BULK_BYTES)),
            int(msg.get("streams", 1)),
        )
        self.send({"kind": "PROBE_RESULT", "rid": msg["rid"], "row": _jsonable(row)})

    def _on_measure(self, msg, body):
        link = self.emu.link_to(msg.get("region"))
        try:
            res = links.measure_pair(
                msg["host"],
                int(msg["port"]),
                link,
                int(msg.get("bulk_bytes", links.PROBE_BULK_BYTES)),
                int(msg.get("streams", 1)),
            )
            self.send({"kind": "MEASURE_RESULT", "rid": msg["rid"], "ok": True, "link": res})
        except ConnectionError as e:
            self.send({"kind": "MEASURE_RESULT", "rid": msg["rid"], "ok": False, "error": str(e)})

    def _on_fit(self, msg, body):
        out = self._run_slot_tool(
            "fit",
            "wantrain.resources.fit",
            ["--rank", str(msg["rank"]), "--peer", str(msg.get("peer", 0))],
            body,
            float(msg.get("timeout_s", 600)),
        )
        res = None
        for line in out.stdout.splitlines():
            if line.startswith("WT_FIT "):
                res = json.loads(line[len("WT_FIT ") :])
        self.send(
            {
                "kind": "FIT_RESULT",
                "rid": msg["rid"],
                "fit": res or {"ok": False, "error": (out.stdout + out.stderr)[-400:]},
            }
        )

    def _on_profile(self, msg, body):
        """Time this slot's F, B and W passes on this node (zero-bubble costs)."""
        out = self._run_slot_tool(
            "profile",
            "wantrain.resources.profile",
            [
                "--rank",
                str(msg["rank"]),
                "--peer",
                str(msg.get("peer", 0)),
                "--reps",
                str(msg.get("reps", 3)),
            ],
            body,
            float(msg.get("timeout_s", 900)),
        )
        res = None
        for line in out.stdout.splitlines():
            if line.startswith("WT_PROFILE "):
                res = json.loads(line[len("WT_PROFILE ") :])
        self.send(
            {
                "kind": "PROFILE_RESULT",
                "rid": msg["rid"],
                "profile": res or {"ok": False, "error": (out.stdout + out.stderr)[-400:]},
            }
        )

    def _run_slot_tool(self, name: str, module: str, argv: list, body: bytes, timeout: float):
        """Run a FIT or PROFILE child on the spec in `body`. An hf spec's child
        starts under a fresh pycache prefix in hfm's scrubbed environment
        (BA-423), the prefix removed at exit; any other starts as before."""
        d = os.path.join(self.workdir, name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"spec-{uuid.uuid4().hex[:8]}.json")
        text = body.decode()
        with open(path, "w") as f:
            f.write(text)
        tail = ["-m", module, "--spec", path, *argv]
        try:
            hf = _is_hf(json.loads(text))
        except ValueError:
            hf = False  # the child reports the malformed spec, as before
        if not hf:
            return subprocess.run(
                [self.python, *tail],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env(),
            )
        from wantrain.model import hfm

        return run_child(
            lambda prefix: [self.python, *hfm.interpreter_args(prefix), *tail],
            hfm.stage_env(self._env()),
            os.path.join(d, "pycache"),
            timeout,
        )

    def _own_devices(self) -> set:
        """The devices a GATE may name: the CPU and this node's slot device."""
        own = {"cpu"}
        dev = self.probe_info.get("device") or self.device
        if isinstance(dev, str) and dev:
            own.add(dev)
        return own

    def _on_gate(self, msg, body, epoch: int | None = None):
        """Run hfm's gate once per requested device, each in its own child under
        a fresh pycache prefix in the stage environment (I-9, BA-428), and reply
        GATE_REPORT. A malformed request, a failed child or an oversized report
        is answered with an ERROR naming the reason."""
        errs = validate_gate_request(msg, self._own_devices())
        if errs:
            raise ValueError("GATE refused: " + "; ".join(errs))
        from wantrain.model import hfm

        if epoch is None:
            epoch = self._gate_children.epoch
        root = os.path.join(self._run_dir(msg["run_id"]), "gate")
        os.makedirs(root, exist_ok=True)
        env = self._env()
        env.pop("WANTRAIN_PEER_KEY")  # the gate never proves the node's key
        env = hfm.stage_env(env)
        reports = []
        for dev in msg["devices"]:
            work = tempfile.mkdtemp(prefix="gate-", dir=root)
            try:
                inp, out = os.path.join(work, "in.json"), os.path.join(work, "out.json")
                with open(inp, "w") as f:
                    json.dump(
                        {"hf_config": msg["hf_config"], "recipe": msg["recipe"], "device": dev}, f
                    )
                res = run_child(
                    lambda prefix, i=inp, o=out: [
                        self.python,
                        *hfm.interpreter_args(prefix),
                        "-m",
                        "wantrain.model.hfm",
                        "gate",
                        "--in",
                        i,
                        "--out",
                        o,
                    ],
                    env,
                    root,
                    GATE_TIMEOUT_S,
                    tracker=self._gate_children,
                    epoch=epoch,
                )
                if self._gate_children.epoch != epoch:
                    raise ChildAborted(f"GATE on {dev} aborted")
                if res.returncode != 0:
                    raise ValueError(
                        f"gate child on {dev} exited {res.returncode}: "
                        f"{(res.stdout + res.stderr)[-400:]}"
                    )
                reports.append(read_gate_report(out, dev))
            finally:
                shutil.rmtree(work, ignore_errors=True)
        self.send(gate_reply(msg.get("rid"), msg["run_id"], reports))

    def _env(self) -> dict:
        from wantrain.runtime.determinism import pinned_env

        env = pinned_env(dict(os.environ))
        env["PYTHONUNBUFFERED"] = "1"
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # The stage proves this agent's own key on the data plane, the one its
        # control handshake proved and the coordinator pinned (CONTROL-I22),
        # never a host-wide key inherited from the environment.
        env["WANTRAIN_PEER_KEY"] = self.key_path
        return env

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self.workdir, "runs", run_id)

    def _on_assign(self, msg, body):
        run_id, attempt = msg["run_id"], int(msg["attempt"])
        rank, peer = int(msg["rank"]), int(msg["peer"])
        spec = json.loads(body.decode())
        run_dir = self._run_dir(run_id)
        # out_dir is a per-node path, not part of the recipe: every byte that
        # decides numerics is the coordinator's, the location is ours.
        spec["out_dir"] = os.path.join(run_dir, "out")
        os.makedirs(spec["out_dir"], exist_ok=True)
        spec_path = os.path.join(run_dir, f"spec-a{attempt}-r{rank}p{peer}.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        with self._lock:
            if self.stage and self.stage.alive():
                self.stage.kill()
            self.assignment = dict(
                run_id=run_id,
                attempt=attempt,
                rank=rank,
                peer=peer,
                tag=spec["tag"],
                out_dir=spec["out_dir"],
                final=bool(msg.get("final")),
            )
            self.last_step = None
            cmd = [
                self.python,
                "-m",
                "wantrain.worker.stage_main",
                "--spec",
                spec_path,
                "--rank",
                str(rank),
                "--peer",
                str(peer),
                # The coordinator checked the stage's device class before
                # ASSIGN, so stage_main may run a pieces spec (BA-58).
                "--launcher",
                "coordinator",
            ]
            log = os.path.join(run_dir, f"stage-r{rank}p{peer}.log")
            env, prefix = self._env(), None
            if _is_hf(spec):
                # BA-423: a fresh pycache prefix per child in hfm's scrubbed
                # environment; StageProcess removes it once the child exits.
                from wantrain.model import hfm

                root = os.path.join(run_dir, "pycache")
                os.makedirs(root, exist_ok=True)
                prefix = tempfile.mkdtemp(prefix="pyc-", dir=root)
                cmd[1:1] = hfm.interpreter_args(prefix)
                env = hfm.stage_env(env)
            self.stage = StageProcess(
                cmd,
                log,
                env,
                lambda line, a=attempt: self._on_stage_line(a, line),
                on_exit=lambda st, a=attempt: self._on_stage_exit(a, st),
                prefix=prefix,
            )
            stage = self.stage
        self.send(
            {
                "kind": "ASSIGN_ACK",
                "rid": msg.get("rid"),
                "run_id": run_id,
                "attempt": attempt,
                "pid": stage.proc.pid,
            }
        )
        threading.Thread(target=self._supervise, args=(stage, attempt), daemon=True).start()

    def _on_stage_line(self, attempt: int, line: str) -> None:
        if line.startswith("WT_WAVE "):
            self._on_wave_line(attempt, line)
            return
        if line.startswith("WT_TELEM "):
            self._forward_telemetry(attempt, line)
            return
        d = None
        if line.startswith("WT_STEP "):
            try:
                d = {"step": int(line.split()[1]), "loss": None, "t": time.time()}
            except (IndexError, ValueError):
                return
            src = "step"
        elif line.startswith("WT_METRIC "):
            try:
                d = json.loads(line[len("WT_METRIC ") :])
            except ValueError:
                return
            src = "metric"
        if d is not None:
            with self._lock:
                a = self.assignment
                if not a or a["attempt"] != attempt:
                    return
                step = d.get("step")
                self.last_step = int(step) if step is not None else None
            self.send(
                {
                    "kind": "PROGRESS",
                    "run_id": a["run_id"],
                    "attempt": attempt,
                    "step": d.get("step"),
                    "loss": d.get("loss"),
                    "t": d.get("t"),
                    "src": src,
                }
            )

    def _on_wave_line(self, attempt: int, line: str) -> None:
        """The stage committed a wave: record it as announced (a ship gap is
        then never silent) and wake the shipper. Shipping happens off the
        stage's stdout reader, so a slow link never blocks the stage."""
        try:
            step = int(line.split()[1])
        except (IndexError, ValueError):
            return
        with self._lock:
            a = self.assignment
            if a and a["attempt"] == attempt and step >= 0:
                self._seen.add((a["run_id"], a["rank"], a["peer"], step))
        self._ship_wake.set()

    def _forward_telemetry(self, attempt: int, line: str) -> None:
        """One live per-step summary to the coordinator, which validates it
        (CONTROL-I18). Anything but a JSON object of at most TELEM_LINE_MAX
        bytes is dropped here; it never travels as TELEMETRY."""
        if len(line.encode()) > TELEM_LINE_MAX:
            return
        try:
            summary = json.loads(line[len("WT_TELEM ") :])
        except ValueError:
            return
        if not isinstance(summary, dict):
            return
        with self._lock:
            a = self.assignment
            if not a or a["attempt"] != attempt:
                return
        self.send(
            {"kind": "TELEM_STEP", "run_id": a["run_id"], "attempt": attempt, "summary": summary}
        )

    def _supervise(self, stage: StageProcess, attempt: int) -> None:
        # CONTROL-I23: every committed wave ships, in step order, woken by each
        # WT_WAVE line and by a backstop scan every pass, and once more at exit.
        # The wake is cleared before the scan, so a wave committed during a
        # scan wakes the next one.
        while stage.alive():
            self._ship_waves(attempt)
            self._ship_wake.wait(SHIP_SCAN_S)
            self._ship_wake.clear()
        rc = stage.wait()
        if rc != 0:
            # Normally already sent by the reaper (_on_stage_exit).
            self._report_exit(stage, attempt, rc)
        self._ship_waves(attempt)
        stage.close()
        if rc == 0:
            if self.sim_exit_delay_s:
                self._stop.wait(self.sim_exit_delay_s)
            # A clean exit reports after its last wave shipped, and its RESULT
            # follows on the same channel.
            a = self._report_exit(stage, attempt, rc)
            if a is not None:
                self._send_results(a)

    def _on_stage_exit(self, attempt: int, stage: StageProcess) -> None:
        """The reaper's call, once the stage ended and its output is drained.
        A stage that died reports at once, even while the supervisor is
        shipping a wave: its peers' "peer lost" exits follow from its death,
        and the judge must hear the cause before them, or it reads a dead
        process as a broken data plane and restarts in place instead of
        convicting it (S2 before S3). Its waves still ship after, and the
        heal's ABORT ships them under the ship lock before its ACK. A clean
        exit is reported by the supervisor, after its last wave."""
        if stage.rc != 0:
            self._report_exit(stage, attempt, stage.rc)
        self._ship_wake.set()

    def _report_exit(self, stage: StageProcess, attempt: int, rc: int) -> dict | None:
        """Send EXIT for `attempt` if it is still the current one, once per
        stage; mark the stage reported. -> the assignment it reported under."""
        with self._lock:
            if stage.exit_claimed:
                return None
            stage.exit_claimed = True
            a = self.assignment
            current = bool(a and a["attempt"] == attempt)
        if not current or a is None:
            stage.reported = True
            return None
        self.send(
            {
                "kind": "EXIT",
                "run_id": a["run_id"],
                "attempt": attempt,
                "rc": rc,
                "done_marker": stage.done_marker,
                "log_tail": stage.tail(),
            }
        )
        stage.reported = True
        return a

    def _send_results(self, a: dict) -> None:
        out = a["out_dir"]
        tag = a["tag"]
        losses = os.path.join(out, f"{tag}_losses.json")
        payload = {
            "kind": "RESULT",
            "run_id": a["run_id"],
            "attempt": a["attempt"],
            "rank": a["rank"],
            "peer": a["peer"],
        }
        if os.path.exists(losses):
            with open(losses) as f:
                payload["losses"] = json.load(f)
        tel = os.path.join(out, f"{tag}_rank{a['rank']}p{a['peer']}_telemetry.jsonl")
        if os.path.exists(tel) and self.ch is not None:
            try:
                self.ch.send_file(
                    {
                        "kind": "TELEMETRY",
                        "run_id": a["run_id"],
                        "attempt": a["attempt"],
                        "rank": a["rank"],
                        "peer": a["peer"],
                    },
                    tel,
                )
            except ChannelClosed:
                pass
        self.send(payload)

    def _ship_waves(self, attempt: int) -> None:
        """Ship every committed wave of the slot not shipped yet, ascending."""
        with self._ship_lock:
            self._ship_waves_locked(attempt)

    def _ship_waves_locked(self, attempt: int) -> None:
        with self._lock:
            a = self.assignment
        if not a or a["attempt"] != attempt or self.ch is None:
            return
        slot = (a["run_id"], a["rank"], a["peer"])
        try:
            on_disk = dict(checkpoint.slot_waves(a["out_dir"], a["tag"], a["rank"], a["peer"]))
        except (ValueError, OSError) as e:
            print(f"[agent {self.name}] wave scan failed: {e}", flush=True)
            return
        newest = max(on_disk, default=None)
        with self._lock:
            self._seen.update(slot + (s,) for s in on_disk)
            steps = sorted(k[3] for k in self._seen if k[:3] == slot)
        for step in steps:
            key = slot + (step,)
            if key in self._shipped or key in self._given_up:
                continue
            d = on_disk.get(step)
            try:
                shipped = d is not None and self._ship_wave(a, attempt, step, d)
            except ChannelClosed:
                return  # the rest ships, in order, on the next connection
            except OSError as e:
                # Retried, in order, on the next pass; never skipped silently.
                print(
                    f"[agent {self.name}] wave r{a['rank']}p{a['peer']} step={step} "
                    f"not shipped yet: {e}",
                    flush=True,
                )
                return
            if not shipped:
                if newest is not None and step < newest and not self._wave_on_disk(a, step):
                    # Pruned (or dropped) before it was shipped: never silent (BA-15).
                    self._given_up.add(key)
                    print(
                        f"[agent {self.name}] ship gap r{a['rank']}p{a['peer']} step={step}",
                        flush=True,
                    )

    @staticmethod
    def _wave_on_disk(a: dict, step: int) -> bool:
        return os.path.isdir(
            os.path.normpath(
                checkpoint.wave_dir(a["out_dir"], a["tag"], a["rank"], a["peer"], step)
            )
        )

    def _ship_wave(self, a: dict, attempt: int, step: int, d: str) -> bool:
        """One committed wave as a unit: a CKPT_PART per file its wave.json
        lists, then the CKPT carrying that manifest; the alias is never read.
        The files are hard-linked aside first, so the stage pruning the wave
        mid-send cannot cut a part short. -> True once the CKPT is sent;
        ChannelClosed propagates."""
        ch = self.ch
        if ch is None:
            raise ChannelClosed("no control connection")
        try:
            m = checkpoint.read_manifest(d, step)
        except (ValueError, OSError):
            return False  # pruned since the scan
        hold = tempfile.mkdtemp(
            prefix=f".ship-r{a['rank']}p{a['peer']}-", dir=self._run_dir(a["run_id"])
        )
        try:
            try:
                for e in m["files"]:
                    _hold_file(os.path.join(d, e["name"]), os.path.join(hold, e["name"]))
            except FileNotFoundError:
                return False  # pruned since the scan
            wave = f"[agent {self.name}] wave r{a['rank']}p{a['peer']} step={step}"
            head = {"run_id": a["run_id"], "attempt": attempt, "step": step}
            for e in m["files"]:
                sent = ch.send_file(
                    dict(head, kind="CKPT_PART", name=e["name"]), os.path.join(hold, e["name"])
                )
                if sent != e["sha256"]:
                    # An immutable wave that does not match its manifest never will.
                    self._given_up.add((a["run_id"], a["rank"], a["peer"], step))
                    print(
                        f"{wave} not shipped: {e['name']} does not match its manifest", flush=True
                    )
                    return False
            ch.send(dict(head, kind="CKPT", manifest=m))
        finally:
            shutil.rmtree(hold, ignore_errors=True)
        self._shipped.add((a["run_id"], a["rank"], a["peer"], step))
        print(f"{wave} shipped", flush=True)
        return True

    def _on_abort(self, msg, body):
        self._gate_children.abort()
        with self._lock:
            stage = self.stage
            a = self.assignment
        if stage:
            stage.kill()
        if a:
            self._ship_waves(a["attempt"])
        with self._lock:
            if self.assignment is a:
                self.assignment = None
        self.send({"kind": "ABORT_ACK", "rid": msg.get("rid")})

    def _on_restore_part(self, msg, path):
        """One file of the wave the next RESTORE commits; held until then."""
        if isinstance(path, str):
            with self._lock:
                self._restore_parts.append((msg, path))

    def _on_restore(self, msg, path):
        """Commit the restored wave (its RESTORE_PARTs, checked against the
        manifest) and point the alias at it, where the engine's resume reads.

        Exclusive: refused while the stage is alive (an ABORT waits for its
        exit) and run under _ship_lock, so no ship spans it. Every received
        body is unlinked on every path, and the restored wave is recorded as
        shipped: the coordinator already holds it (CONTROL-I23)."""
        with self._lock:
            parts, self._restore_parts = self._restore_parts, []
        try:
            with self._ship_lock:
                error = self._restore_locked(msg, parts)
        finally:
            for p in [p for _, p in parts] + ([path] if isinstance(path, str) else []):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass  # moved into the wave
        if error is not None:
            print(
                f"[agent {self.name}] restore r{msg.get('rank')}p{msg.get('peer')} "
                f"step={str(msg.get('step'))[:40]} refused: {error}",
                flush=True,
            )
        self.send(
            {"kind": "RESTORE_ACK", "rid": msg.get("rid"), "ok": error is None, "error": error}
        )

    def _restore_locked(self, msg, parts) -> str | None:
        """-> None once the wave is committed and the alias names it, else why not."""
        step = msg.get("step")
        if type(step) is not int:
            return f"malformed step {str(step)[:40]!r}"
        with self._lock:
            stage, a = self.stage, self.assignment
        if stage is not None and stage.alive():
            who = f"r{a['rank']}p{a['peer']} attempt {a['attempt']} " if a else ""
            return f"refused: stage {who}(pid {stage.proc.pid}) is still alive"
        run_id, tag, rank, peer = msg["run_id"], msg["tag"], int(msg["rank"]), int(msg["peer"])
        manifest = msg.get("manifest")
        try:
            files = checkpoint.check_manifest(manifest, step, f"restore r{rank}p{peer} step {step}")
        except ValueError as e:
            return f"refused: {e}"
        got: dict = {}
        for pm, p in parts:
            if (pm.get("run_id"), pm.get("rank"), pm.get("peer"), pm.get("step")) == (
                run_id,
                rank,
                peer,
                step,
            ):
                got.setdefault(pm.get("name"), (pm, p))
        out = os.path.join(self._run_dir(run_id), "out")
        os.makedirs(out, exist_ok=True)
        staged = tempfile.mkdtemp(prefix=".restore-", dir=self._run_dir(run_id))
        try:
            for e in files:
                if e["name"] not in got:
                    return f"refused: no RESTORE_PART for {e['name']}"
                pm, p = got[e["name"]]
                if pm.get("sha256") != e["sha256"]:
                    return (
                        f"sha256 mismatch ({e['name']}: part {str(pm.get('sha256'))[:12]}, "
                        f"manifest {e['sha256'][:12]})"
                    )
                os.replace(p, os.path.join(staged, e["name"]))
            try:
                d, committed = checkpoint.commit_received_wave(
                    out, tag, rank, peer, manifest, staged
                )
                if not committed:  # an equal wave was held: kept untouched, the alias moved to it
                    checkpoint.link_legacy(d, out, tag, rank, peer)
            except ValueError as e:
                return f"refused: {e}"
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        self._shipped.add((run_id, rank, peer, step))
        return None

    def _on_release(self, msg, body):
        with self._lock:
            if self.stage and self.stage.alive():
                self.stage.kill()
            self.stage = None
            self.assignment = None
        self.send({"kind": "RELEASE_ACK", "rid": msg.get("rid")})

    def _on_shutdown(self, msg, body):
        self._stop.set()
        self._shutdown()
        if self.ch:
            self.ch.close()


def _hold_file(src: str, dst: str) -> None:
    """dst names src's bytes for as long as dst exists: a hard link (a copy
    where links are refused). FileNotFoundError when src is gone."""
    try:
        os.link(src, dst)
    except FileNotFoundError:
        raise
    except OSError:
        shutil.copyfile(src, dst)


def _jsonable(row: dict) -> dict:
    def fix(v):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return None
        return v

    return {k: [fix(x) for x in vals] for k, vals in row.items()}
