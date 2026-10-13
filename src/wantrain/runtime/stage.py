"""Pipeline stage runtime: zero-bubble pipeline parallelism, one path.

Each slot (stage, peer) executes, every step, the precomputed order of F, B
and W passes its spec carries (scheduler/zero_bubble.py) for the microbatches
it owns. F runs the stage forward; B computes only the input gradient, sends
it upstream at once (the critical path) and leaves each linear layer's weight
gradient pending (model/wgrad.py); W runs those deferred dW GEMMs where the
schedule placed them, inside what would otherwise be bubbles.

Topology: a width-1 chain uses one authenticated duplex TCP connection per
hop (optionally several parallel streams); a stage of w > 1 peers shares its
microbatches by weighted ownership (spec.mb_owner), its peers form a clique
for the ZeRO-1 optimizer exchange, and adjacent stages connect peer to peer.

Exactness contract: per-microbatch weight gradients are folded in fixed
microbatch-index order whatever order they are produced in
(runtime/accumulate.py), so the schedule, jitter and ownership cannot change
the bits.
"""

import base64
import contextlib
import hashlib
import itertools
import json
import math
import os
import queue
import re
import shutil
import socket
import sys
import threading
import time

import torch

from wantrain.fault_tolerance.checkpoint import (
    WAVE_STATE,
    commit_wave,
    drop_waves_after,
    slot_waves,
    validate_wave,
    wave_dir,
)
from wantrain.model import provision, wgrad
from wantrain.model.data import make_batch
from wantrain.model.gpt import build_layers, loss_fn
from wantrain.runtime import exchange as xc
from wantrain.runtime import logical_ckpt as lc
from wantrain.runtime import optim
from wantrain.runtime import pieces as pc
from wantrain.runtime.accumulate import CountFold, OrderedAccumulator, TiedRowFold
from wantrain.runtime.determinism import float_hex, setup_determinism
from wantrain.runtime.spec import JobSpec, first_pin_difference
from wantrain.runtime.telemetry import parse_sim_slow, summarize_step
from wantrain.transport import framing as fr
from wantrain.transport import serial, tuning
from wantrain.transport.shim import DelayedDelivery, LinkSpec, SenderPacer, SharedBudget


def engine_supplies_autocast(model) -> bool:
    """Whether the engine opens the autocast region, or the model does itself.
    The built-in architectures autocast inside their own forward, so the engine
    must leave them alone or their bitwise baselines move; an operator's model
    and HF's modules (BA-409) have no such code and get the contract from here.
    fp32 needs no region."""
    return getattr(model, "arch", None) in ("user", "hf") and model.autocast_enabled()


def compute_autocast(model, device_type: str, enabled: bool = None):
    """The region the layers run in: a real nullcontext when not needed, not
    `torch.autocast(enabled=False)`, which actively disables autocast and would
    make the built-in blocks re-enable it inside themselves, giving the right
    result by cancellation rather than by leaving them alone."""
    if enabled is None:
        enabled = engine_supplies_autocast(model)
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=model.compute_torch_dtype(), enabled=True)


HELLO = fr.HELLO
# One WT_TELEM line, prefix included, never exceeds this (the agent drops longer).
TELEM_LINE_MAX = 4096
# Control frames carry objects through transport/serial.py, never pickle.
OBJ = "obj"
# How long shutdown waits for queued sends to drain and for every neighbour to
# say it is done, before closing anyway (a wedged peer must not pin a box).
SHUTDOWN_GRACE_S = 120.0
# Bound on each read of a data-plane handshake and its HELLO: a socket that
# sends nothing within it is refused, never waited on (RUNTIME-I16).
HANDSHAKE_TIMEOUT_S = 30.0


class LineAtomicStdout:
    """A stdout that hands each complete line to the wrapped stream in one
    write, under one lock. print() writes a line's text and its newline in two
    calls, so the loss tail's WT_METRIC (another thread, stage_main) could glue
    onto this loop's WT_TELEM or WT_STEP, and the agent drops a glued line:
    a lost final WT_STEP turns the shutdown quiet back into a stall
    (RUNTIME-I9, CONTROL-I17). Each thread buffers its own partial line; flush()
    also writes the caller's partial line, as the wrapped stream would."""

    def __init__(self, out):
        self._out = out
        self._lock = threading.Lock()
        self._local = threading.local()

    def write(self, s):
        buf = getattr(self._local, "buf", "") + s
        cut = buf.rfind("\n") + 1
        self._local.buf = buf[cut:]
        if cut:
            with self._lock:
                self._out.write(buf[:cut])
        return len(s)

    def flush(self):
        buf, self._local.buf = getattr(self._local, "buf", ""), ""
        with self._lock:
            if buf:
                self._out.write(buf)
            self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)


def frag_size(nbytes: int, flows: int, lo: int = 64 << 10, hi: int = 1 << 20) -> int:
    """TRANSPORT-I12: the fragment size that spreads one message over every flow of a hop: nbytes / flows, within
    [lo, hi]. A fixed 1 MB cut a 2.15 MB boundary into three fragments, so three of sixteen flows carried each
    message and the rest idled; TCP restarts an idle flow's window, and each 1 MB fragment then took several round
    trips at 141 ms (2026-10-13, NC-SE: 37% of stage 0's time in waits over 0.5 s). The bytes and their order are
    unchanged: the receiver reassembles by index."""
    if flows <= 1:
        return hi
    return max(lo, min(hi, -(-nbytes // flows)))


class _WTask:
    """One microbatch's streamed W (RUNTIME-I21): wgrad.flush_each's gradients (its ops, its bits), each folded as
    soon as it is complete and dropped; B's were folded at B, so the fold is bit-identical to _store_grads' add of
    the whole set. RUNTIME-I27: it can stop between gradients at a deadline and resume where it stopped; the
    gradients, their order and their folds are the same however it is split.

    One module-level class whose instances drop every reference when they finish: a class built per task is a
    reference cycle (a type refers to itself), which only the cyclic collector frees, and each one held its step's
    accumulator, a full set of fp32 gradients (~1.45 GiB a step on a 0.42B-parameter stage, 2026-10-13)."""

    __slots__ = ("stage", "step", "mb", "vs", "acc", "index", "n", "it", "spent")

    def __init__(self, stage, step, mb, vs, params, store):
        self.stage, self.step, self.mb, self.vs = stage, step, mb, vs
        self.acc = stage._step_acc(step, vs)
        self.index = {id(p): i for i, p in enumerate(params)}
        self.n = len(params)
        self.it = wgrad.flush_each(store, widen=False)  # RUNTIME-I28: widened by the fold
        self.spent = 0.0

    def run(self, deadline=None) -> bool:
        if self.it is None:
            return True
        a, mb, n, index = self.acc, self.mb, self.n, self.index
        t0 = time.monotonic()
        for p, g in self.it:
            i = index.get(id(p))
            if i is not None:
                a.fold_part(mb, i, n, g, p.dtype)
            del g
            if deadline is not None and time.monotonic() >= deadline:
                self.spent += time.monotonic() - t0
                return False
        self.spent += time.monotonic() - t0
        k = a.order.index(mb)
        if n and (
            a.next is None or any((mb, i) not in a.parts and a.next[i] <= k for i in range(n))
        ):
            raise RuntimeError("a trainable parameter received no gradient in B/W")
        a.end_part(mb)
        stage, step, vs = self.stage, self.step, self.vs
        stage.done_mbs.setdefault(step, set()).add((mb, vs))
        tw = time.monotonic() - self.spent
        self.stage = self.acc = self.it = self.index = None  # nothing of this step outlives the W
        stage._sim_slow_wait(step, tw)
        stage._ev("bwd_w", step, mb, vs, tw)
        return True

    def __call__(self):
        self.run(None)


class StageRuntime:
    # The checkpoint commit in flight (_checkpoint, _await_wave).
    _wave_commit = None
    # ADR-0006 modes (set in __init__; class defaults keep a bare runtime zero1).
    native = False  # wire "native": W streams gradients in their production dtype
    pieces_mode = False  # shard "pieces": peer k owns piece k of every trainable tensor
    copy: xc.CopyEngine | None = None  # the copy engine, when this slot exchanges with peers
    _budgets: dict = {}
    _seen: set | frozenset = frozenset()  # check_frame's duplicate keys (a set in __init__)
    # The steps run() ran and their wall time, for WT_AEAD (BA-44).
    _steps_run = 0
    _steps_wall_ns = 0
    # M4 (set in __init__; class defaults keep a bare runtime dense, untied).
    hf = False  # arch "hf": HF units (RUNTIME-I19)
    moe = False  # MoE layers on this slot (RUNTIME-I18)
    _relay = False  # a tied table split across stages, relayed in TIED frames
    _tied_role = None  # "embed" | "head" | "mid" on a relay plan
    _tied: torch.nn.Parameter | None = None  # the tied table on an end stage of the relay
    _tied_keys: list = []  # stage 0's state_dict keys of its working copy
    _biases: list = []  # MoE selection biases, in layer order
    _bias_names: list = []  # shard "pieces": the biases' chunk names
    _bias_layer: dict = {}  # shard "pieces": bias name -> MoE layer
    _net_logged = False  # WT_NET dial printed (TRANSPORT-I10)
    _opt_thread: threading.Thread | None = None  # RUNTIME-I23: the step's optimizer, overlapped
    _opt_ready: list | None = (
        None  # per chunk layer: set once its parameters hold the step's update
    )
    _opt_error: list = []
    _opt_logged = False
    _opt_steps = (
        0  # optimizer steps this process has started (RUNTIME-I24 promotes after the second)
    )
    _opt_events: list | None = (
        None  # per chunk layer: the CUDA event its update ends with (RUNTIME-I23)
    )
    _opt_stream: torch.cuda.Stream | None = (
        None  # the optimizer's side CUDA stream, one per run (RUNTIME-I23)
    )

    def __init__(self, spec: JobSpec, rank: int, peer: int = 0):
        self.spec = spec.finalize()
        self.rank = rank  # stage index
        self.peer = peer  # peer index inside the stage
        self.S = spec.stages
        self.V = spec.n_virtual()
        self.M = spec.microbatches
        self.widths = spec.stage_widths
        self.w = self.widths[rank]
        self.wide = any(x > 1 for x in self.widths)
        self.my_vs = [rank]
        self.final_vs = self.S - 1
        self.hosts_final = rank == self.S - 1
        self.my_mbs = spec.owned(rank, peer)
        self.fwd_origin = {}  # (step, mb) -> upstream peer that sent FWD
        # fixed-order incremental accumulation (no O(M) grad retention):
        self.acc = {}  # (step, vs) -> OrderedAccumulator (chain path)
        self.shard_acc = {}  # step -> OrderedAccumulator (wide owner, stream)
        self.done_mbs = {}  # step -> {(mb, vs)} executed locally

        setup_determinism(spec.init_seed)
        self.device = torch.device(spec.device_for(rank, peer))
        # The memory free before this stage built anything: the base of the fit check's budget, which the moments'
        # residency (_promote_moments) keeps to.
        self._free_at_init = (
            int(torch.cuda.mem_get_info(self.device)[0]) if self.device.type == "cuda" else 0
        )
        self.hf = spec.model.arch == "hf"
        if self.hf:
            full = self._build_hf()
        else:
            # RUNTIME-I22: only this stage's entries stay resident (one stage per rank).
            full = build_layers(spec.model, spec.init_seed, span=tuple(spec.partition[rank]))
        self.chunks = {}
        for vs in self.my_vs:
            a, b = spec.partition[vs]
            self.chunks[vs] = torch.nn.ModuleList(full[a:b]).to(self.device)
        if self.hf:
            self._hf_start_asserts(full)
        self._init_tied(full)

        # Parameters whose dW the split linears defer (model/wgrad.py), per chunk.
        self._wgrad_owned = {vs: wgrad.owned_params(self.chunks[vs]) for vs in self.my_vs}

        # Open the autocast region here only for an operator's model; built-in
        # models open it in their own forward (engine_supplies_autocast). Without
        # it a user model's bf16 activations meet fp32 parameters: a dtype crash
        # at best, silent fp32 compute reported as bf16 at worst.
        self._user_autocast = engine_supplies_autocast(spec.model)
        # Only trainable parameters enter the pipeline's parameter list: it
        # feeds the optimizer, the ordered accumulator and the ZeRO-1 shard
        # split alike, and a frozen (LoRA base) weight that slipped through
        # would be charged two Adam moments it never uses.
        self.params = []
        for vs in self.my_vs:
            self.params += self._chunk_params(vs)
        if self._tied_role == "head":
            self._tied_idx = next(i for i, p in enumerate(self.params) if p is self._tied)
        if self.wide and self.w > 1:
            # ZeRO-1: optimizer state exists only for the owned shard
            self.shard_params = self.params[self.peer :: self.w]
        else:
            self.shard_params = self.params
        self._init_moe()
        self.native = spec.wire == "native"
        self.pieces_mode = spec.shard == "pieces"
        # Timing-only seams (BA-38): they move when bytes go, never a byte.
        self._fifo = os.environ.get("WANTRAIN_TEST_QUEUE") == "fifo"
        self._barrier_always = os.environ.get("WANTRAIN_TEST_BARRIER") == "always"
        self._updates_between_ops = os.environ.get("WANTRAIN_TEST_UPDATES") == "between_ops"
        # E4's timing-only seams (exchange chunk, intra one-way delay, slot padding).
        self._chunk_bytes = xc.chunk_bytes_seam(os.environ.get("WANTRAIN_TEST_CHUNK_BYTES"))
        self._intra_ms = intra_latency_seam(os.environ.get("WANTRAIN_TEST_INTRA_LATENCY_MS"))
        self._slot_pad = slot_pad_seam(os.environ.get("WANTRAIN_TEST_SLOT_PAD"), rank)
        self._slot_of = {mb: i for i, mb in enumerate(self.my_mbs)}
        self._pad_t0 = None  # (step, when its slot-0 F began, _gather_wait then)
        self._gather_wait = 0.0  # seconds _await_layer waited, summed
        # Exchange state shared by wire "native" and shard "pieces".
        self.holds = xc.HoldCounters()
        self._seq = itertools.count()
        self._parts: dict = {}  # (kind, step, mb, unit, sender) -> received chunks
        self._seen = set()  # check_frame's duplicate keys
        self._exec_step = 0  # the step _run_step executes (check_frame's step, zero1)
        if self.pieces_mode:
            self._init_pieces()
            self.opt = None
        elif not self.shard_params:
            # A ZeRO-1 peer of a stage wider than its trainable tensors owns none
            # (M2B-ZERO1-EMPTY-SHARD): no optimizer. It still ships its gradient
            # slices, installs every owner's PARAMS broadcast and checkpoints an empty
            # optimizer state (_opt_state). At width 1 the same holds for stage 0 of
            # a split tied table whose only parameter is the table the Head owns.
            self.opt = None
            self._init_zero1_tables()
        else:
            self.opt = optim.build(spec.optimizer, self.shard_params, spec.lr)
            self._init_zero1_tables()
        if self.w > 1 and (self.native or self.pieces_mode):
            self.copy = xc.CopyEngine(self.device)

        self.inbox = queue.Queue()
        # Set by _finish before sockets come down, so connection errors during
        # a normal shutdown are not reported as a dead peer.
        self._closing = False
        self.fwd_ready = {}
        self.bwd_ready = {}
        self.ctx = {}
        self.deferred_w = []
        self._w_slices = 0  # RUNTIME-I27: times a filling W stopped between gradients (WT_WFILL)
        self.grads = {}
        self.losses = {}
        # Appended by sender threads and the main loop alike; _ev and
        # _take_events hold the lock, so no event is lost or taken twice.
        self.telemetry = []
        self._tel_lock = threading.Lock()
        self._emit_us = 0.0  # duration of the previous WT_TELEM block
        # Timing-only test seam: slow this slot's F/B/W by a factor.
        slow = parse_sim_slow(os.environ.get("WANTRAIN_SIM_SLOW"))
        self._sim_slow = (
            (slow[2], slow[3]) if slow is not None and slow[:2] == (rank, peer) else None
        )
        self._sync_grads = {}  # from_peer -> {mb: [shard grad tensors]}
        self._sync_params = {}  # owner_peer -> [tensors]
        self._peer_losses = []  # LOSS payloads collected at final peer 0

        # socket registries
        self.conns = {}  # ('next',q) ('prev',q) ('intra',p); key + (i,) per extra stream
        # Every sender-side thread (encoders included), so _finish can join
        # them after the shutdown sentinel instead of guessing with a sleep.
        self._send_threads = []
        # Every reader thread, so shutdown can wait for each neighbour's END
        # before closing: closing a socket the peer is still writing to
        # resets the peer's connection, which it would rightly read as a crash.
        self._read_threads = []
        self.send_qs = {}  # same keys -> queue for the sender thread
        # Every bound SecureSocket, whose seal/open counters WT_AEAD sums.
        self._secure = []

    # ---------------- HF units, tied tables, MoE: setup (M4) ----------------

    def _build_hf(self):
        """arch "hf" (RUNTIME-I19, BA-406, BA-449): the environment checked
        (hfm.guard_environment, never set), then the units. A checkpoint run
        loads only this stage's units (span); a from-config run builds in full
        (ADR-0009 decision 7; hfm ignores the span), then the run's buffer
        record is applied, refused naming a tensor whose bytes differ."""
        from wantrain.model import hfm

        hfm.guard_environment()
        a, b = self.spec.partition[self.rank]
        full = build_layers(self.spec.model, self.spec.init_seed, span=(a, b))
        record = self.spec.model.hf_buffers
        if not isinstance(record, dict):
            raise RuntimeError("arch 'hf' needs model.hf_buffers (the run's buffer record, I-4)")
        hfm.apply_buffer_record(getattr(full, "root", full), record)
        return full

    def _hf_start_asserts(self, full):
        """BA-406, BA-444, BA-448: after the build, the determinism read-back
        and the HF flags the units run under; a failure is fatal, naming the
        flag."""
        from wantrain.model import hfm
        from wantrain.runtime.determinism import check_determinism

        check_determinism()
        where = f"stage r{self.rank}p{self.peer}"

        def fail(flag, why):
            raise RuntimeError(f"HF-ASSERT {flag}: {why} ({where})")

        root = getattr(full, "root", full)
        cfg = getattr(root, "config", None)
        if cfg is None:
            fail("config", "the units' root carries no HF config")
        attn = getattr(cfg, "_attn_implementation", None)
        if attn != "eager":
            fail("_attn_implementation", f"is {attn!r}, the run needs 'eager'")
        for key, v in sorted(cfg.to_dict().items()):
            dropout = "dropout" in key or key.endswith("pdrop")
            if dropout and isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
                fail(key, f"is {v!r}, every dropout must be 0")
        for name, m in root.named_modules():
            if isinstance(m, torch.nn.Dropout) and m.p != 0:
                fail("dropout", f"module {name or '<root>'} has p={m.p}")
            if getattr(m, "gradient_checkpointing", False) is True:
                fail("gradient_checkpointing", f"is on in module {name or '<root>'}")
        if getattr(root, "is_gradient_checkpointing", False):
            fail("gradient_checkpointing", "is on in the model")
        if getattr(cfg, "use_cache", None) is not False:
            fail("use_cache", f"is {getattr(cfg, 'use_cache', None)!r}, the run needs False")
        for vs in self.my_vs:
            for name, m in self.chunks[vs].named_modules():
                if not m.training:
                    fail("training", f"module {name or '<chunk>'} is in eval mode")
        have = hfm.hf_code()
        if have != self.spec.model.hf_code:
            fail("hf_code", f"the installed code {have!r} differs from the pin")

    def _init_tied(self, full):
        """RUNTIME-I19: a tied table split across stages (spec.tied_relay()).
        The Head stage owns the table (in self.params, its AdamW and the row
        fold); stage 0 holds a working copy of the step's rows, NaN elsewhere,
        out of self.params, the optimizer, the fold and the checkpoint; the
        stages between forward TIED fragments from their reader threads."""
        self._relay = self.hf and self.spec.tied_relay()
        self._tied_role = None
        self._tied = None
        self._tied_keys: list = []
        self._tfold: dict = {}  # step -> TiedRowFold (Head)
        # stage 0: step -> {k: (row ids, values)}; on a multi-stream link the
        # frames of one step may complete in any order
        self._tied_rows: dict = {}
        self._tinst: tuple = (None, set())  # (step, k installed) of stage 0's working copy
        self._tied_rmemo: dict = {}  # step -> [R_k] (make_batch once per step, not per frame)
        self._tied_go = threading.Event()  # set once the TIED sequences exist
        self._tied_n: dict = {}  # src -> TIED frames counted by its readers (END check)
        self._tied_frags: dict = {}  # (src, step, mb) -> fragments seen of a frame
        self._tied_socks: dict = {}  # src -> sockets of a multi-stream link (one END each)
        self._tied_ends: dict = {}  # src -> ENDs seen
        self._tied_lock = threading.RLock()  # _tied_R takes it inside _forward_tied
        if not self._relay:
            return
        lo, hi = self.spec._tied_stages()
        if self.rank == lo:
            self._tied_role = "embed"
        elif self.rank == hi:
            self._tied_role = "head"
        elif lo < self.rank < hi:
            self._tied_role = "mid"
            return
        else:
            return
        mine = {id(p) for vs in self.my_vs for p in self.chunks[vs].parameters()}
        far = full[-1] if self._tied_role == "embed" else full[0]
        shared = [p for p in far.parameters() if id(p) in mine]
        if len(shared) != 1:
            raise RuntimeError(
                f"tied relay: stage {self.rank} shares {len(shared)} parameters with the far "
                f"unit, the plan needs exactly one tied table"
            )
        self._tied = shared[0]
        if self._tied_role == "embed":
            sd = self.chunks[self.my_vs[0]].state_dict(keep_vars=True)
            self._tied_keys = [k for k, v in sd.items() if v is self._tied]

    def _init_moe(self):
        """RUNTIME-I18: the MoE modules of this slot (every module that calls
        moe_forward carries moe_layer, I-11), their selection biases, and the
        ZeRO-1 ownership list: the trainable parameters, then the biases in
        layer order (bias j at index n_params + j, owner index % w). A dense
        stage has none and its lists are unchanged."""
        mods = [
            m for vs in self.my_vs for m in self.chunks[vs].modules() if hasattr(m, "moe_layer")
        ]
        self.moe = bool(mods)
        self._moe_layers = sorted({int(m.moe_layer) for m in mods})
        bias = {
            int(m.moe_layer): m
            for m in mods
            if isinstance(getattr(m, "sel_bias", None), torch.Tensor)
        }
        self._bias_mods = [bias[layer] for layer in sorted(bias)]
        self._biases = [m.sel_bias for m in self._bias_mods]
        self._own = self.params + self._biases
        w = self.w if self.wide and self.w > 1 else 1
        n = len(self.params)
        self._my_bias = [
            j for j in range(len(self._biases)) if (n + j) % w == (self.peer if w > 1 else 0)
        ]
        self.moe_aux: list = []  # [[step, mb, layer, hex]] of the microbatches this slot owns
        self._moe_f: dict = {}  # (step, mb) -> {layer: (hex, counts)} from F
        self._moe_counts: dict = {}  # step -> {mb: {layer: counts}}
        self._cfold: dict = {}  # step -> CountFold (the owner of some biases)

    def _moe_recipe(self, mod):
        """The module's own MoE recipe (arch.moe_recipe of its moe_cfg: a
        GPTConfig for a native block, the row's recipe dict for a swapped HF
        block, whose top-k and bias rate are not the JobSpec model's)."""
        from wantrain.model import arch

        return arch.moe_recipe(getattr(mod, "moe_cfg", None) or self.spec.model)

    def _moe_k(self, mod) -> int:
        return int(self._moe_recipe(mod).moe_top_k)

    def _count_fold(self, step) -> CountFold:
        f = self._cfold.get(step)
        if f is None:
            mods = [self._bias_mods[j] for j in self._my_bias]
            t_mb = self.spec.mb_size * self.spec.model.seq_len
            f = self._cfold[step] = CountFold(
                step,
                self.M,
                [int(m.moe_layer) for m in mods],
                {int(m.moe_layer): int(m.sel_bias.numel()) for m in mods},
                {int(m.moe_layer): t_mb * self._moe_k(m) for m in mods},
            )
        return f

    def _moe_layer_bias(self, layer) -> int:
        return next(j for j, m in enumerate(self._bias_mods) if int(m.moe_layer) == layer)

    def _count_add(self, step, mb, counts, sender):
        """Fold one microbatch's count vectors (a dict {layer: int64 (E,)})
        received from `sender`, or this slot's own, at the bias owner."""
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"MOE-COUNT (step {step}, mb {mb}, peer {sender}): counts are not a dict"
            )
        if not counts and not self._my_bias:
            return  # a peer owning no bias receives empty count dicts
        fold = self._count_fold(step)
        for layer, vec in counts.items():
            fold.add(mb, layer, vec, sender)

    def _counts_for(self, step, mb, owner) -> dict:
        """This slot's count vectors of microbatch mb for the biases `owner`
        owns (ZeRO-1 ownership list)."""
        w = self.w if self.wide and self.w > 1 else 1
        n = len(self.params)
        rec = self._moe_counts.get(step, {}).get(mb)
        if rec is None:
            raise RuntimeError(f"MOE-COUNT (step {step}, mb {mb}): no counts recorded in F")
        out = {}
        for j, m in enumerate(self._bias_mods):
            if (n + j) % w == owner:
                out[int(m.moe_layer)] = rec[int(m.moe_layer)]
        return out

    def _record_moe(self, step, mb, vs):
        """F's records (RUNTIME-I18): each MoE layer's aux hex and int64
        counts, once per (step, mb, layer)."""
        from wantrain.model import arch

        got = moe_records(arch.take_moe_records(self.chunks[vs]), step, mb, self._moe_layers)
        if (step, mb) in self._moe_f:
            raise RuntimeError(f"MOE-RECORD (step {step}, mb {mb}): recorded twice")
        self._moe_f[(step, mb)] = got
        if self._biases:  # counts feed only the selection biases (none: nothing to hold)
            self._moe_counts.setdefault(step, {})[mb] = {layer: c for layer, (_, c) in got.items()}
        self.moe_aux += [[step, mb, layer, h] for layer, (h, _) in sorted(got.items())]

    def _replay_moe(self, step, mb, vs):
        """B's rebuild recomputed the layers: its records must equal F's."""
        from wantrain.model import arch

        got = moe_records(arch.take_moe_records(self.chunks[vs]), step, mb, self._moe_layers)
        moe_replay_check(step, mb, self._moe_f.pop((step, mb)), got)

    def _moe_bias_step(self, step, chunk):
        """After the step's AdamW: the owned biases move by the step's counts
        (arch.moe_bias_update; the counts folded by integer add in microbatch
        order, CountFold). Width 1 folds its own records."""
        if not self._my_bias:
            self._moe_counts.pop(step, None)
            return
        from wantrain.model import arch

        if not (self.wide and self.w > 1):
            for mb in range(self.M):
                self._count_add(step, mb, self._counts_for(step, mb, 0), self.peer)
        counts = self._count_fold(step).result()
        self._cfold.pop(step)
        tokens = self.M * self.spec.mb_size * self.spec.model.seq_len
        with torch.no_grad():
            arch.moe_bias_update(chunk, counts, tokens)
        self._moe_counts.pop(step, None)

    # ---------------- tied table relay (RUNTIME-I19, BA-411..BA-416) ----------------

    def _tied_ids(self, step, mb) -> torch.Tensor:
        """T_mb: the microbatch's sorted unique ids, from make_batch on this
        stage (no index travels)."""
        return torch.unique(self._batch(step, mb)[0].reshape(-1), sorted=True)

    def _tied_R(self, step) -> list:
        """[R_k] for k = 0..M-1: R_k = T_k minus every T_j, j < k, ascending.
        Computed once per step (a receipt needs one R_k per frame or fragment),
        under the relay lock (readers of a multi-stream link share it); steps
        before step - 1 are dropped."""
        with self._tied_lock:
            got = self._tied_rmemo.get(step)
            if got is not None:
                return got
            seen = torch.zeros(self.spec.model.vocab_size, dtype=torch.bool)
            out = []
            for k in range(self.M):
                t = self._tied_ids(step, k)
                r = t[~seen[t]]
                seen[r] = True
                out.append(r)
            for old in [s_ for s_ in self._tied_rmemo if s_ < step - 1]:
                del self._tied_rmemo[old]
            self._tied_rmemo[step] = out
            return out

    def _tied_fold(self, step) -> TiedRowFold:
        f = self._tfold.get(step)
        if f is None:
            assert self._tied is not None
            f = self._tfold[step] = TiedRowFold(self._tied.shape[0], range(self.M))
        return f

    def _tied_plan(self, start):
        """The TIED sequences of this attempt, once its first step is known:
        forward TIED(t, i) for every step run, i in stage 0's B order; backward
        TIED(t, k), k = 0..M-1, for every step whose rows come from the Head
        (from step 1, or from the first step of a resume, whose rows the Head
        sends from the loaded table)."""
        steps, lo = self.spec.steps, self.spec._tied_stages()[0]
        b_order = [int(mb) for op, mb in self.spec.schedule_order[lo][0] if op == "B"]
        first_bwd = start if self.spec.resume else start + 1
        self._tseq = {
            "fwd": _TiedSeq([(t, i) for t in range(start, steps) for i in b_order]),
            "bwd": _TiedSeq([(t, k) for t in range(first_bwd, steps) for k in range(self.M)]),
        }
        self._first_bwd = first_bwd
        self._tied_go.set()

    def _tied_ok(self, step, mb) -> bool:
        """Stage 0's F of microbatch mb waits for R_0..R_mb of its step."""
        if self._tied_role != "embed" or step < self._first_bwd:
            return True
        got = self._tied_rows.get(step, {})
        return all(k in got for k in range(mb + 1))

    def _tied_install(self, step):
        """Stage 0's working copy for `step`: NaN, then the rows received so far
        (at the attempt's first step of a fresh launch, the built table's rows
        of the step's ids)."""
        t = self._tied
        with torch.no_grad():
            if self._tinst[0] != step:
                if step < self._first_bwd:
                    keep = torch.zeros(t.shape[0], dtype=torch.bool)
                    for k in range(self.M):
                        keep[self._tied_ids(step, k)] = True
                    t[(~keep).to(t.device)] = float("nan")
                else:
                    t.fill_(float("nan"))
                self._tinst = (step, set())
                for old in [s_ for s_ in self._tied_rows if s_ < step]:
                    del self._tied_rows[old]
            done = self._tinst[1]
            for k, (idx, vals) in sorted(self._tied_rows.get(step, {}).items()):
                if k not in done:
                    t[idx.to(t.device)] = vals.to(t.device, t.dtype)
                    done.add(k)

    def _send_tied_embed(self, step, mb, e):
        """After B_i on stage 0: TIED-SPARSE, then TIED(t, i) = e_i[T_i]."""
        if e is None:
            raise RuntimeError(f"TIED (step {step}, mb {mb}): the tied table got no gradient")
        ids = self._tied_ids(step, mb).to(e.device)
        out = torch.ones(e.shape[0], dtype=torch.bool, device=e.device)
        out[ids] = False
        check_tied_sparse(step, mb, e, out)
        self.send_qs[("next", 0)].put(
            (fr.TIED, step, mb, 0, e[ids].detach().float().contiguous().cpu(), "raw")
        )

    def _send_tied_rows(self, step):
        """The Head, after the table's AdamW (or a resume's load): TIED(step, k)
        = R_k of the table, k = 0..M-1, toward stage 0."""
        table = self._tied.detach()
        for k, r in enumerate(self._tied_R(step)):
            rows = table[r.to(table.device)].float().contiguous().cpu()
            self.send_qs[("prev", 0)].put((fr.TIED, step, k, 0, rows, "raw"))

    def _on_tied(self, f, src):
        """A whole TIED frame at an end of the relay, validated before its
        payload is used: a plan with a relay (its rollback read at receipt,
        BA-467), the direction, the sequence, the codec and shape; then folded
        (Head) or held for the F it feeds."""
        piece = self.spec._tied_rolled_back() if self._relay else None
        if piece is not None:
            raise RuntimeError(
                f"refused TIED frame (step {f.step}, mb {f.mb}) from {src}: piece {piece!r} is "
                f"rolled back to 'tied-split-refused'"
            )
        if not self._relay or self._tied_role not in ("embed", "head"):
            raise RuntimeError(
                f"refused TIED frame (step {f.step}, mb {f.mb}) from {src}: this plan has no "
                f"tied relay at stage {self.rank}"
            )
        head = self._tied_role == "head"
        want = ("prev", 0) if head else ("next", 0)
        if src != want:
            raise RuntimeError(f"refused TIED frame (step {f.step}, mb {f.mb}) from {src}")
        self._tseq["fwd" if head else "bwd"].push(f.step, f.mb, 0, 1)
        rows = self._tied_ids(f.step, f.mb) if head else self._tied_R(f.step)[f.mb]
        d = int(self._tied.shape[1])
        if f.codec != fr.RAW_F32 or tuple(f.shape) != (int(rows.numel()), d):
            raise RuntimeError(
                f"refused TIED frame (step {f.step}, mb {f.mb}): codec {f.codec} shape "
                f"{tuple(f.shape)}, want RAW_F32 ({rows.numel()}, {d})"
            )
        t = f.tensor()
        if head:
            self._tied_fold(f.step).add_embed(f.mb, t.to(self.device))
        else:
            self._tied_rows.setdefault(f.step, {})[f.mb] = (rows, t)

    def _forward_tied(self, f, src) -> bool:
        """A stage between stage 0 and the Head: each TIED fragment leaves on
        the other side as it arrives, from the reader thread, byte for byte,
        checked first (codec, shape, (step, mb) sequence). -> False when it is
        refused (the fatal is in the inbox)."""
        self._tied_go.wait()
        fwd = src[0] == "prev"
        try:
            piece = self.spec._tied_rolled_back()
            if piece is not None:
                raise RuntimeError(f"piece {piece!r} is rolled back to 'tied-split-refused'")
            with self._tied_lock:
                self._tseq["fwd" if fwd else "bwd"].push(f.step, f.mb, f.frag_idx, f.frag_cnt)
                rows = (self._tied_ids(f.step, f.mb) if fwd else self._tied_R(f.step)[f.mb]).numel()
                want = (int(rows), int(self.spec.model.d_model))
                if f.codec != fr.RAW_F32 or tuple(f.shape) != want:
                    raise RuntimeError(
                        f"TIED fragment codec {f.codec} shape {tuple(f.shape)}, want RAW_F32 {want}"
                    )
                q = self.send_qs[("next", 0) if fwd else ("prev", 0)]
                q.put((fr.TIED, f.step, f.mb, f.chunk, _RawFrag(f), None))
        except (RuntimeError, IndexError, KeyError) as e:
            self.inbox.put(("fatal", f"refused TIED (step {f.step}, mb {f.mb}) from {src}: {e}"))
            return False
        return True

    def _tied_end(self, src):
        """END on a relay socket (reader thread): every TIED frame of the
        attempt that comes through it must have arrived; a missing one is fatal
        (a whole frame missing at an end, a fragment missing at a stage
        between)."""
        if not self._relay or self._tied_role is None or not self._tied_go.is_set():
            return
        with self._tied_lock:
            # a multi-stream link closes with one END per socket: the frames
            # are all in only once the last one has arrived
            ends = self._tied_ends[src] = self._tied_ends.get(src, 0) + 1
        if ends < self._tied_socks.get(src, 1):
            return
        seq = self._tseq["fwd" if src[0] == "prev" else "bwd"]
        if self._tied_role == "mid":
            short = not seq.done()
        else:
            short = self._tied_n.get(src, 0) < len(seq.expected)
        if short:
            self.inbox.put(
                ("fatal", f"TIED frames missing from {src} at its END (expected {seq.left()})")
            )

    def _drain_tied_relay(self):
        """A stage between stage 0 and the Head forwards TIED frames from its
        reader threads into its send queues, and its own loop can end before
        the last of them arrive (stage 0's last forward rows, the Head's last
        backward rows). The shutdown sentinel would then reach the senders
        first and their END would overtake those frames, which the far end
        refuses as missing. So the sentinel waits until every TIED frame of
        the attempt has been forwarded in both directions; the inbox is pumped
        meanwhile, so a refusal or a lost neighbour stays fatal, and the wait
        is bounded like the rest of the shutdown."""
        if self._tied_role != "mid" or not self._relay or not self._tied_go.is_set():
            return
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        while True:
            with self._tied_lock:
                left = {d: q.left() for d, q in self._tseq.items() if not q.done()}
            if not left:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"TIED relay: frames still expected after {SHUTDOWN_GRACE_S:.0f} s at "
                    f"shutdown (first missing per direction: {left})"
                )
            self._pump(block=True, timeout=0.05)

    def _tied_seen(self, f, src) -> bool:
        """Reader thread, before reassembly: a TIED fragment at a stage between
        is forwarded (-> True, consumed); at an end, a whole frame is counted
        for the END check."""
        if self._tied_role == "mid" and self._relay:
            if not self._forward_tied(f, src):
                raise _TiedRefused()
            return True
        key = (src, f.step, f.mb)
        with self._tied_lock:
            got = self._tied_frags[key] = self._tied_frags.get(key, 0) + 1
            if got >= f.frag_cnt:
                del self._tied_frags[key]
                self._tied_n[src] = self._tied_n.get(src, 0) + 1
        return False

    # ---------------- exact sharded stages: setup (ADR-0006) ----------------

    def _init_zero1_tables(self):
        """check_frame's plan tables for a ZeRO-1 stage: per stride param i the
        production dtype of its gradient (wgrad.gather_dtypes: the compute
        dtype for split linears under autocast under wire "native", else the
        parameter's dtype) and, for the fp32 wire's whole-shard frames, the
        parameter dtypes of each peer's stride."""
        if not (self.wide and self.w > 1):
            return
        chunk = self.chunks[self.my_vs[0]]
        m = self.spec.model
        gd = wgrad.gather_dtypes(chunk, m.compute_dtype, m.autocast_enabled())
        self._pindex = {id(p): i for i, p in enumerate(self.params)}
        self._grad_dt = [gd[id(p)] if self.native else p.dtype for p in self.params]
        self._native_acc: dict = {}  # (step, i) -> OrderedAccumulator (wire "native")
        self._native_table = {
            i: [(self._grad_dt[i], int(self.params[i].numel()))]
            for i in range(self.peer, len(self.params), self.w)
        }
        # RUNTIME-I18: one COUNT unit per owned selection bias (n_params + j).
        n = len(self.params)
        for j in self._my_bias:
            self._native_table[n + j] = [(torch.int64, int(self._biases[j].numel()), xc.COUNT_LANE)]

    def _init_pieces(self):
        """Peer k owns piece k (pieces.piece_layout, or spec.piece_bounds) of
        every trainable tensor: an fp32 master, AdamW m and v, an fp32 fold
        accumulator. Every peer holds a working copy of every tensor in the
        dtype its consumer reads (wgrad.gather_dtypes, MODEL-I7), in the
        module parameter itself, tagged per piece with the step it serves."""
        spec = self.spec
        chunk = self.chunks[self.my_vs[0]]
        m = spec.model
        named = [(n, p) for n, p in chunk.named_parameters() if p.requires_grad]
        self.pnames = [n for n, _ in named]
        # MoE selection biases (buffers) ride their block's small-tensor unit:
        # their "gradient" is the int64 count lane, their update the bias rule
        # (RUNTIME-I18, BA-455); no AdamW state, no logical piece of their own.
        units = pc.exchange_units(chunk)
        trainable = {id(p) for _, p in named}
        biases = [(n, t) for u in units for n, t in u if id(t) not in trainable]
        self._bias_names = [n for n, _ in biases]
        by_id = {id(mod.sel_bias): int(mod.moe_layer) for mod in self._bias_mods}
        self._bias_layer = {n: by_id[id(t)] for n, t in biases}
        named = named + biases
        self._pname = {id(p): n for n, p in named}
        self.pby = dict(named)
        self.units = [[n for n, _ in u] for u in units]
        self.unit_of = {n: u for u, names in enumerate(self.units) for n in names}
        uid = {id(self.pby[n]): u for n, u in self.unit_of.items()}
        # Units each layer reads (the tied table: the Head reads the Embed's unit).
        self.layer_units = [
            sorted({uid[id(p)] for p in layer.parameters() if id(p) in uid}) for layer in chunk
        ]
        self.shapes = {n: tuple(p.shape) for n, p in named}
        bounds = (spec.piece_bounds or [None] * self.S)[self.rank]
        self.layout = pc.piece_layout(self.shapes, self.w, bounds)
        gd = wgrad.gather_dtypes(chunk, m.compute_dtype, m.autocast_enabled())
        self.gdt = {n: gd.get(id(p), p.dtype) for n, p in named}
        with torch.no_grad():
            self.master = {
                n: pc.piece(p.detach(), self.layout[n], self.peer).to(torch.float32, copy=True)
                for n, p in named
            }
            for n, p in named:
                if p.dtype != self.gdt[n]:
                    p.data = p.data.to(self.gdt[n])
        self.popt = pc.adamw([self.master[n] for n in self.pnames], spec.lr)
        self._pfold: dict = {}  # (step, unit) -> _UnitFold
        self._ready: list = []  # (step, unit) whose fold completed, not yet updated
        self._upd_done: dict = {}  # step -> units updated
        self._ran_step = -1  # last step whose ops all ran
        self._step_t0: dict = {}  # step -> perf_counter at its start
        # Piece sizes per owner and the plan tables check_frame reads.
        size = {n: [b[k + 1] - b[k] for k in range(self.w)] for n, b in self.layout.items()}
        self._grads_table = {
            u: [
                (torch.int64, size[n][self.peer], xc.COUNT_LANE)
                if n in self._bias_layer
                else (self.gdt[n], size[n][self.peer])
                for n in names
            ]
            for u, names in enumerate(self.units)
        }
        self._params_tables = [
            {u: [(self.gdt[n], size[n][k]) for n in names] for u, names in enumerate(self.units)}
            for k in range(self.w)
        ]

    def _init_copies(self, start):
        """Every working copy serves step `start` (built from the seed, or from
        the resumed logical state)."""
        self.piece_tag = [[start] * self.w for _ in self.units]
        self.copy_got = {(u, start): self.w for u in range(len(self.units))}
        self._upd_step = start
        self._next_report = start

    # ---------------- exact sharded stages: exchange (ADR-0006) ----------------

    def _send_exchange(self, ftype, kind, dest, unit, step, mb, tensors):
        """Queue one unit's frame for intra peer `dest`: chunks of at most 16
        MiB (WANTRAIN_TEST_CHUNK_BYTES: smaller, timing only; unit, offset and
        length in the OBJ envelope, BA-13, BA-40), each copied D2H by the copy
        engine, served need-first (RUNTIME-I13)."""
        hold = "host_grads" if ftype == fr.GRADS else "gathers_out"
        q = self.send_qs[("intra", dest)]
        for env, view in xc.chunk_object(unit, tensors, self._chunk_bytes):
            nb = _nbytes(view)
            self.holds.add(hold, nb)
            key = xc.NeedKey(unit, kind, step, mb, next(self._seq))
            body = _Exchange(env, self.copy.d2h(view), nb, hold, key)
            q.push(key, (ftype, step, mb, self.peer, body, OBJ))

    def _h2d(self, tensors):
        """Received host tensors on this slot's device, copied by the copy
        engine (the compute thread only waits for it)."""
        if self.copy is None:
            return list(tensors)
        return self.copy.h2d(tensors).result()

    @staticmethod
    def _refuse_multi_use(store):
        """Wire "native" ships a split linear's gradient when its last
        contribution is computed; a module used twice per microbatch would sum
        in its working dtype instead of the reference's fp32: refused loudly."""
        mods = [id(m) for m, _ in store]
        if len(set(mods)) != len(mods):
            raise RuntimeError(
                "wire 'native' does not support a split linear used twice in one microbatch"
            )

    def _w_native(self, step, mb, vs, params, store, other_grads):
        """W under wire "native" on a wide ZeRO-1 stage: the gradients B made
        (embeddings, norms) ship when W starts, then each split linear's as
        flush_iter yields it, in its production dtype; the stride owner widens
        each contribution to the parameter dtype before the ordered fold."""
        self._refuse_multi_use(store)
        self.done_mbs.setdefault(step, set()).add((mb, vs))
        shipped = 0
        for p in params:
            if id(p) in other_grads:
                self._ship_native(step, mb, p, other_grads[id(p)])
                shipped += 1
        for p, g in wgrad.flush_iter(store, self.spec.emission_order[self.rank]):
            self._ship_native(step, mb, p, g)
            shipped += 1
        if shipped != len(params):
            raise RuntimeError("a trainable parameter received no gradient in B/W")
        if self._biases:
            # RUNTIME-I18 (BA-454): one COUNT frame per (mb, bias), RAW_I64 (E,).
            n = len(self.params)
            for k in range(self.w):
                for layer, vec in self._counts_for(step, mb, k).items():
                    j = self._moe_layer_bias(layer)
                    if k == self.peer:
                        self._count_add(step, mb, {layer: vec}, self.peer)
                    else:
                        self._send_exchange(
                            fr.GRADS, xc.COUNT, k, n + j, step, mb, [vec.reshape(-1)]
                        )

    def _ship_native(self, step, mb, p, g):
        if g is None:
            raise RuntimeError("a trainable parameter received no gradient in B/W")
        i = self._pindex[id(p)]
        owner = i % self.w
        if owner == self.peer:
            self._native_fold(step, i, mb, g)
        else:
            self._send_exchange(fr.GRADS, xc.GRAD, owner, i, step, mb, [g.reshape(-1)])

    def _native_fold(self, step, i, mb, g):
        a = self._native_acc.get((step, i))
        if a is None:
            a = self._native_acc[(step, i)] = OrderedAccumulator(range(self.M))
        p = self.params[i]
        a.add(mb, [g.to(p.dtype).reshape(p.shape)])

    def _w_pieces(self, step, mb, params, store, other_grads):
        """W under shard "pieces": every unit (pieces.exchange_units) is
        emitted as soon as all its gradients exist: the B-made ones (the
        Embed's tables, norms, the tied table) when W starts, a split linear's
        when flush_iter yields it. Piece k of each goes to peer k; this peer's
        own piece goes straight into its fold."""
        self._refuse_multi_use(store)
        got: dict = {}
        left = [len(u) for u in self.units]

        def take(p, g):
            if g is None:
                raise RuntimeError("a trainable parameter received no gradient in B/W")
            n = self._pname[id(p)]
            got[n] = g
            u = self.unit_of[n]
            left[u] -= 1
            if left[u] == 0:
                self._emit_unit(step, mb, u, [got.pop(x) for x in self.units[u]])

        for p in params:
            if id(p) in other_grads:
                take(p, other_grads[id(p)])
        if self._bias_names:
            counts = self._moe_counts.get(step, {}).get(mb)
            if counts is None:
                raise RuntimeError(f"MOE-COUNT (step {step}, mb {mb}): no counts recorded in F")
            for n in self._bias_names:
                take(self.pby[n], counts[self._bias_layer[n]])
            per_step = self._moe_counts[step]
            del per_step[mb]
            if not per_step:
                del self._moe_counts[step]
        for p, g in wgrad.flush_iter(store, self.spec.emission_order[self.rank]):
            take(p, g)
        if any(left):
            raise RuntimeError("a trainable parameter received no gradient in B/W")

    def _emit_unit(self, step, mb, u, grads):
        names = self.units[u]
        flat = [g.reshape(-1) for g in grads]
        for k in range(self.w):
            part = [f[self.layout[n][k] : self.layout[n][k + 1]] for f, n in zip(flat, names)]
            if k == self.peer:
                self._fold_piece(step, u, mb, part, own=True)
            else:
                self._send_exchange(fr.GRADS, xc.GRAD, k, u, step, mb, part)

    def _fold_piece(self, step, u, mb, tensors, own):
        f = self._pfold.get((step, u))
        if f is None:
            f = self._pfold[(step, u)] = _UnitFold(self.M, self.holds, self._h2d)
        if f.add(mb, tensors, own):
            self._ready.append((step, u))

    def _run_updates(self):
        """RUNTIME-I13: every unit whose fold completed is updated now: AdamW on
        the owned piece, the cast to the consumer dtype, the gather to every
        peer; several at once go in (step, forward unit) order."""
        if not self._ready:
            return
        ready, self._ready = sorted(self._ready), []
        for step, u in ready:
            self._update_unit(step, u)

    def _update_unit(self, step, u):
        t0 = time.monotonic()
        names = self.units[u]
        grads = self._pfold.pop((step, u)).acc
        tn = [i for i, n in enumerate(names) if n not in self._bias_layer]
        if tn:
            pc.step_unit(self.popt, [self.master[names[i]] for i in tn], [grads[i] for i in tn])
        for i, n in enumerate(names):
            if n in self._bias_layer:
                # after the unit's AdamW: the bias rule on this piece, from the
                # step's int64 counts folded in microbatch order (RUNTIME-I18)
                self._bias_rule(n, grads[i])
        copies = []
        with torch.no_grad():
            for n in names:
                wc = pc.cast_piece(self.master[n], self.gdt[n])
                pc.piece(self.pby[n].data, self.layout[n], self.peer).copy_(wc)
                copies.append(wc)
        self._installed(u, self.peer, step + 1)
        for k in range(self.w):
            if k != self.peer:
                self._send_exchange(fr.PARAMS, xc.GATHER, k, u, step + 1, 0, copies)
        self._ev("step_opt", step, -1, -1, t0, unit=u)
        self._upd_done[step] = self._upd_done.get(step, 0) + 1
        while self._upd_done.get(self._upd_step, 0) >= len(self.units):
            self._upd_step += 1
            cut = self._upd_step - 1
            self._seen = {k for k in self._seen if k[1] >= cut}

    def _bias_rule(self, n, counts):
        """sel_bias += gamma * sign(T_g * k - E * c) on this peer's piece
        (SPEC-model §2's selection-bias update, elementwise): T_g the step's
        tokens, E the layer's experts, k its top-k."""
        mod = next(m for m in self._bias_mods if int(m.moe_layer) == self._bias_layer[n])
        r = self._moe_recipe(mod)
        gamma = float(r.moe_bias_rate)
        if gamma == 0:
            return
        e, k = int(r.moe_experts), int(r.moe_top_k)
        t_g = self.M * self.spec.mb_size * self.spec.model.seq_len
        sgn = torch.sign(t_g * k - e * counts.to(torch.int64))
        master = self.master[n]
        # arch.moe_bias_update's exact expression: a move of +gamma, -gamma or
        # none (never an add of 0.0)
        moved = master + sgn.to(master.device, torch.float32) * gamma
        master.copy_(torch.where(sgn.to(master.device) == 0, master, moved))

    def _installed(self, u, k, s):
        """Piece k of unit u now serves step s."""
        self.piece_tag[u][k] = s
        self.copy_got[(u, s)] = self.copy_got.get((u, s), 0) + 1
        self.copy_got.pop((u, s - 2), None)

    def _install_gather(self, s, u, k, tensors):
        ts = self._h2d(tensors)
        with torch.no_grad():
            for n, t in zip(self.units[u], ts):
                pc.piece(self.pby[n].data, self.layout[n], k).copy_(t)
        self._installed(u, k, s)

    def _await_layer(self, step, li):
        """RUNTIME-I13: an op of `step` reads layer li only once every unit the
        layer reads holds the copy of that step (pumping meanwhile, so updates
        run inside the wait); RUNTIME-I10: a piece tagged with another step is
        fatal."""
        for u in self.layer_units[li]:
            if self.copy_got.get((u, step), 0) < self.w:
                t0 = time.monotonic()
                while self.copy_got.get((u, step), 0) < self.w:
                    self._pump(block=True)
                self._ev("stall", step, -1, -1, t0, wait="gather")
                self._gather_wait += time.monotonic() - t0
            tags = self.piece_tag[u]
            if any(t != step for t in tags):
                raise RuntimeError(
                    f"stale working copy: an op of step {step} reads unit {u} whose pieces "
                    f"serve steps {tags}"
                )

    def _expect(self, kind, sender):
        if self.pieces_mode:
            units = self._grads_table if kind == "GRADS" else self._params_tables[sender]
            cur = self._upd_step
        else:
            units = self._native_table
            cur = self._exec_step
        return {"units": units, "M": self.M, "step": cur, "seen": self._seen}

    def _on_exchange(self, f, msg, src):
        """A received GRADS or PARAMS frame: validated at the boundary before
        any use (exchange.check_frame, BA-12), then reassembled, folded or
        installed. A malformed frame is fatal, naming its class."""
        kind = "GRADS" if f.type == fr.GRADS else "PARAMS"
        if src is None or src[0] != "intra":
            raise xc.FrameError("envelope", f"{kind} frame on a non-intra connection {src}")
        sender = src[1]
        if not (self.native or self.pieces_mode) or (kind == "PARAMS" and not self.pieces_mode):
            self._on_legacy(kind, f, msg, sender)
            return
        if not (
            isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "x" and isinstance(msg[1], dict)
        ):
            raise xc.FrameError(
                "envelope", f"{kind} frame from intra{sender} is not an exchange chunk"
            )
        _, env, tensors = msg
        fenv = dict(env, mb=f.mb, step=f.step, sender=sender)
        xc.check_frame(kind, fenv, tensors, self._expect(kind, sender))
        hold = "host_grads" if kind == "GRADS" else "gathers_in"
        self.holds.add(hold, _nbytes(tensors))
        key = (kind, f.step, f.mb, env["u"], sender)
        parts = self._parts.setdefault(key, [])
        parts.append((env, tensors))
        whole = xc.reassemble(parts)
        if whole is None:
            return
        del self._parts[key]
        u = env["u"]
        if kind == "GRADS" and self.pieces_mode:
            self._fold_piece(f.step, u, f.mb, whole, own=False)
        elif kind == "GRADS" and u >= len(self.params):
            self.holds.sub(hold, _nbytes(whole))
            layer = int(self._bias_mods[u - len(self.params)].moe_layer)
            self._count_add(f.step, f.mb, {layer: whole[0]}, sender)
        elif kind == "GRADS":
            ts = self._h2d(whole)
            self.holds.sub(hold, _nbytes(whole))
            self._native_fold(f.step, u, f.mb, ts[0])
        else:
            self._install_gather(f.step, u, sender, whole)
            self.holds.sub(hold, _nbytes(whole))

    def _on_legacy(self, kind, f, msg, sender):
        """The ZeRO-1 whole-shard frames (wire "fp32" GRADS, and PARAMS under
        "zero1"): each validated by check_frame against the stride's dtypes
        and shapes before it is used."""
        if kind == "PARAMS":
            if not (isinstance(msg, tuple) and len(msg) == 2 and msg[0] == sender):
                raise xc.FrameError("envelope", f"PARAMS from intra{sender} names another owner")
            self._legacy_check(kind, f.step, 0, msg[1], self._own[sender :: self.w], sender)
            self._sync_params[sender] = msg[1]
            return
        mine = self.params[self.peer :: self.w]
        n_mb = 4 if self._biases else 3  # an MoE stage's message carries its counts
        if isinstance(msg, tuple) and len(msg) == n_mb and msg[0] == "mb":
            mb, tensors = msg[1], msg[2]
            self._legacy_check(kind, f.step, mb, tensors, mine, sender)
            if self._biases:
                self._count_add(f.step, mb, msg[3], sender)
            self._shard_add(f.step, mb, tensors)
        elif isinstance(msg, dict):
            msg = dict(msg)
            counts = msg.pop("counts", None) if self._biases else None
            if self._biases and not isinstance(counts, dict):
                raise RuntimeError(
                    f"MOE-COUNT (step {f.step}, peer {sender}): GRADS without its counts"
                )
            for mb, tensors in msg.items():
                self._legacy_check(kind, f.step, mb, tensors, mine, sender)
            for mb, c in (counts or {}).items():
                if mb not in msg:
                    raise RuntimeError(
                        f"MOE-COUNT (step {f.step}, mb {mb}, peer {sender}): counts of a "
                        f"microbatch the message carries no gradient of"
                    )
                self._count_add(f.step, mb, c, sender)
            for mb in msg:
                if counts is not None and mb not in counts:
                    raise RuntimeError(
                        f"MOE-COUNT (step {f.step}, mb {mb}, peer {sender}): count vectors missing"
                    )
            self._sync_grads[len(self._sync_grads)] = msg
        else:
            raise xc.FrameError("envelope", f"GRADS from intra{sender} is not a shard frame")

    def _legacy_check(self, kind, step, mb, tensors, plist, sender):
        if not isinstance(tensors, list) or any(not isinstance(t, torch.Tensor) for t in tensors):
            raise xc.FrameError("envelope", f"{kind} from intra{sender} carries no tensor list")
        if len(tensors) != len(plist):
            raise xc.FrameError("shape", f"{kind} from intra{sender}: {len(tensors)} tensors")
        n = [int(p.numel()) for p in plist]
        env = {"u": "zero1", "off": 0, "len": sum(n), "n": n, "t": list(range(len(n)))}
        env.update(mb=mb, step=step, sender=sender)
        table = {"zero1": [(p.dtype, k) for p, k in zip(plist, n)]}
        exp = {"units": table, "M": self.M, "step": self._exec_step, "seen": self._seen}
        xc.check_frame(kind, env, [t.reshape(-1) for t in tensors], exp)
        if any(t.shape != p.shape for t, p in zip(tensors, plist)):
            raise xc.FrameError("shape", f"{kind} from intra{sender} is not the stride's shapes")

    def _is_barrier(self, step):
        """Shard "pieces" keeps the full step barrier at checkpoint steps and
        the last step only (WANTRAIN_TEST_BARRIER=always: every step, timing
        only)."""
        ck = self.spec.ckpt_every
        return (
            self._barrier_always or step == self.spec.steps - 1 or bool(ck and (step + 1) % ck == 0)
        )

    def _step_complete(self, s):
        return s <= self._ran_step and self._upd_done.get(s, 0) >= len(self.units)

    def _pieces_barrier(self, step):
        n = len(self.units)
        while not (
            self._step_complete(step)
            and all(self.copy_got.get((u, step + 1), 0) >= self.w for u in range(n))
        ):
            self._pump(block=True, top=True)

    def _emit_ready_reports(self):
        """WT_TELEM t, WT_HOLD t and WT_STEP t, in increasing step order, once
        every owned step-t update is applied (BA-10); a barrier step is
        reported by the step loop after its barrier."""
        while (
            self._next_report < self.spec.steps
            and not self._is_barrier(self._next_report)
            and self._step_complete(self._next_report)
        ):
            self._report_step(self._next_report)

    def _report_step(self, step, checkpoint=False):
        t_step = self._step_t0.pop(step)
        self._emit_step_telemetry(step, (time.perf_counter() - t_step) * 1e3, upto=step)
        self.holds.end_step(step)
        print(self.holds.line(step), flush=True)
        self._print_mem(step)
        if checkpoint:
            self._checkpoint(step)
        print(f"WT_STEP {step}", flush=True)
        self._steps_run += 1
        self._steps_wall_ns = time.monotonic_ns() - self._t_first
        self._next_report = step + 1

    # ---------------- networking ----------------

    def _port(self, stage, peer):
        if self.spec.endpoints is not None:
            return self.spec.endpoints[stage][peer][1]
        return self.spec.port_base + stage * 8 + peer

    def _listen_port(self, stage, peer):
        if self.spec.endpoints is not None:
            ep = self.spec.endpoints[stage][peer]
            return ep[2] if len(ep) > 2 else ep[1]
        return self.spec.port_base + stage * 8 + peer

    def _host(self, stage, peer):
        if self.spec.endpoints is not None:
            return self.spec.endpoints[stage][peer][0]
        return self.spec.host

    def _connect(self):
        if self.spec.endpoints is not None and self.spec.slot_peers is None:
            # A coordinator-run spec names every slot's key; without the pins
            # any key the trusted set holds could take any slot (RUNTIME-I16).
            raise ConnectionError(
                "unpinned data plane: a spec with endpoints must carry slot_peers"
            )
        if self.S == 1 and self.w == 1:
            return
        from wantrain.transport.identity import default_key_path
        from wantrain.transport.secure import hardware_aes

        self._keyp = default_key_path()
        self._aes_hw = hardware_aes()
        if self.wide:
            self._connect_mesh()
        else:
            self._connect_chain()

    def _slot_id(self, st, pe):
        """The peer id pinned on slot (st, pe), or None in unpinned local mode."""
        sp = self.spec.slot_peers
        return None if sp is None else sp[st][pe]

    def _admissible(self, pid, want):
        """A slot's authenticated id: the pinned one when pinned, else (local
        mode without slot_peers) any id of the trusted set, if one is set."""
        if want is not None:
            return pid == want
        tp = self.spec.trusted_peers
        return tp is None or pid in tp

    def _seal(self, sock, auth):
        from wantrain.transport.secure import SecureSocket

        return SecureSocket(
            sock,
            auth["cipher"],
            auth["send_key"],
            auth["recv_key"],
            auth["send_prefix"],
            auth["recv_prefix"],
        )

    def _bound(self, hop, ss, auth, st, pe):
        """One legitimate socket, authenticated, sealed and bound to its slot:
        exactly one p2p_auth and one p2p_connected event (RUNTIME-I16)."""
        self._secure.append(ss)
        info = dict(peer=auth["peer_id"], mode=auth["mode"], cipher=auth["cipher"], hop=hop)
        now = time.monotonic()
        self._ev(
            "p2p_auth", 0, -1, -1, now, streams=max(1, self.spec.p2p_streams), stages=self.S, **info
        )
        self._ev("p2p_connected", 0, -1, -1, now, st=st, pe=pe, **info)
        print(f"WT_P2P_BOUND hop={hop} st={st} pe={pe} peer={auth['peer_id']}", flush=True)

    def _dial_secure(self, stage, peer, hello=False):
        """Dial slot (stage, peer), authenticate it as slot_peers[stage][peer]
        (when pinned) and seal it; a mesh dial sends its HELLO right after
        (BA-27). The target is fixed, so any failure here is fatal."""
        from wantrain.transport.identity import authenticate

        sock = self._dial(stage, peer)
        sock.settimeout(HANDSHAKE_TIMEOUT_S)
        try:
            auth = authenticate(
                sock,
                self._keyp,
                trusted_peers=self.spec.trusted_peers,
                expect_peer=self._slot_id(stage, peer),
                aes_hw=self._aes_hw,
            )
            ss = self._seal(sock, auth)
            if hello:
                fr.send_frame(ss, HELLO, 0, 0, (self.rank, self.peer))
            sock.settimeout(None)
        except Exception as e:
            sock.close()
            raise ConnectionError(f"p2p dial to stage {stage} peer {peer} refused: {e}") from e
        return ss, auth

    def _auth_accepted(self, sk):
        """Authenticate an accepted socket with no id expectation (the caller
        checks the id against the slot), mapping every failure to a refusal."""
        from wantrain.transport.identity import authenticate

        try:
            return authenticate(sk, self._keyp, aes_hw=self._aes_hw)
        except TimeoutError:
            raise _Refused("silent") from None
        except Exception as e:  # noqa: BLE001 -- untrusted bytes: any failure is a refusal
            # (json.loads raises RecursionError on a deeply nested message).
            raise _Refused("auth") from e

    def _refuse(self, sk, r):
        print(f"WT_P2P_REFUSED reason={r.reason} st={r.st} pe={r.pe} peer={r.peer}", flush=True)
        self._ev(
            "p2p_refused",
            0,
            -1,
            -1,
            time.monotonic(),
            reason=r.reason,
            st=r.st,
            pe=r.pe,
            peer=r.peer,
        )
        _hard_close(sk)

    def _accept_until_bound(self, lst, expected, admit, dials, hop_of):
        """RUNTIME-I16 acceptor. Each accepted socket is authenticated in its
        own thread under HANDSHAKE_TIMEOUT_S. admit(sock) returns (key, value,
        st, pe, peer) for a legitimate socket (key None: any free key) or
        raises _Refused. A refused socket is logged and closed and never takes
        a key; accepting goes on until every key of `expected` is bound, then
        the listener and every socket still pending are closed. A failed dial
        in `dials` ends the wait with its error. hop_of(key) labels a bound
        socket's events. -> {key: value}"""
        bound = {}
        lock = threading.Lock()
        pending = set()
        all_bound = threading.Event()

        def handle(sk):
            try:
                try:
                    try:
                        # A peer that resets before this point makes these
                        # fail (EINVAL on macOS): a refusal, never a crash.
                        sk.settimeout(HANDSHAKE_TIMEOUT_S)
                        sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except OSError:
                        raise _Refused("auth") from None
                    key, value, st, pe, pid = admit(sk)
                except _Refused as r:
                    if all_bound.is_set() and r.reason in _IO_REASONS:
                        r.reason = "unbound"
                    raise
                with lock:
                    if all_bound.is_set():
                        raise _Refused("unbound", st, pe, pid)
                    if key is None:
                        key = next(k for k in expected if k not in bound)
                    if key in bound:
                        raise _Refused("duplicate", st, pe, pid)
                    # Blocking from here on: the run waits on its peers as long
                    # as the work takes (set before the run can see the socket).
                    sk.settimeout(None)
                    bound[key] = value
                    pending.discard(sk)
                    if len(bound) == len(expected):
                        all_bound.set()
                self._bound(hop_of(key), *value, st, pe)
            except _Refused as r:
                with lock:
                    pending.discard(sk)
                self._refuse(sk, r)

        if not expected:
            all_bound.set()
        lst.settimeout(0.2)
        try:
            while not all_bound.is_set():
                for d in dials:
                    d.raise_if_failed()
                try:
                    sk, _ = lst.accept()
                    tuning.pace(
                        sk
                    )  # TRANSPORT-I13: this side's sends are paced too, inherited or not
                except TimeoutError:
                    continue
                except OSError:
                    # A connection reset before accept (ECONNABORTED) or no
                    # descriptor left (EMFILE) under a flood: an intruder must
                    # not end the stage; back off briefly and keep accepting.
                    all_bound.wait(0.05)
                    continue
                with lock:
                    pending.add(sk)
                threading.Thread(target=handle, args=(sk,), daemon=True).start()
        finally:
            lst.close()
            all_bound.set()
            with lock:
                rest = list(pending)
            for sk in rest:
                # shutdown wakes the handler blocked in recv (close alone
                # does not); the handler then refuses it as unbound.
                with contextlib.suppress(OSError):
                    sk.shutdown(socket.SHUT_RDWR)
        return bound

    def _wire_shared(self, keys, sock, link: LinkSpec):
        """Wire several logical directions onto one socket: both send queues
        drain through a single sender and one reader demuxes by frame type.
        Full-duplex TCP carries forward and backward over the same link, so
        only one dial direction has to succeed."""
        for k in keys:
            self.conns[k] = sock
        q = queue.Queue()
        for k in keys:
            self.send_qs[k] = q
        pacer = SenderPacer(link)
        delay = DelayedDelivery(link, self.inbox.put)
        self._start_reader(self._reader, (sock, delay, keys[0]))
        t = threading.Thread(target=self._sender, args=(sock, q, pacer, _hop(keys[0])), daemon=True)
        t.start()
        self._send_threads.append(t)

    def _connect_chain(self):
        """Pipeline chain over published endpoints, for any stage count: rank r
        dials r+1 and accepts r-1 on its own port; rank 0 accepts nothing, rank
        S-1 dials nothing. No wrap link on purpose: the orchestrator's
        reachability gate only proves the inbound path of ranks 1..S-1, so a
        wrap dial would spin in _dial forever. Ranks launch in descending order,
        so every dial's listener is already up. With p2p_streams > 1, fragments
        round-robin across parallel flows per direction, each direction with
        its own reassembler (exact bytes by construction).
        """
        s = self.spec
        n = max(1, s.p2p_streams)
        has_next = self.rank < self.S - 1
        has_prev = self.rank > 0
        lst = self._listen() if has_prev else None
        # Each stream is dialed, authenticated as slot_peers[rank+1][0] and
        # sealed on its own thread; the acceptor side runs at the same time,
        # since neighbours reach their handshakes at different times.
        dials = (
            [_Result(lambda: self._dial_secure(self.rank + 1, 0)) for _ in range(n)]
            if has_next
            else []
        )
        accepted = {}
        if has_prev:
            want = self._slot_id(self.rank - 1, 0)

            def admit(sk):
                auth = self._auth_accepted(sk)
                pid = auth["peer_id"]
                if not self._admissible(pid, want):
                    raise _Refused("peer", self.rank - 1, 0, pid)
                return None, (self._seal(sk, auth), auth), self.rank - 1, 0, pid

            accepted = self._accept_until_bound(
                lst, list(range(n)), admit, dials, lambda k: "prev0"
            )
        next_socks = []
        for d in dials:
            ss, auth = d.get()
            self._bound("next0", ss, auth, self.rank + 1, 0)
            next_socks.append(ss)
        prev_socks = [accepted[i][0] for i in range(n)] if has_prev else []

        if has_next:
            hop_out = self._link(s.links[self.rank], self.rank + 1, 0)
            if n == 1:
                self._wire_shared([("next", 0)], next_socks[0], hop_out)
            else:
                self._wire_multi([("next", 0)], next_socks, hop_out)
        if has_prev:
            hop_in = self._link(s.links[self.rank - 1], self.rank - 1, 0)
            if n == 1:
                self._wire_shared([("prev", 0)], prev_socks[0], hop_in)
            else:
                self._wire_multi([("prev", 0)], prev_socks, hop_in)

    MULTI_FRAG = 1 << 20  # the largest fragment across parallel streams
    # TRANSPORT-I12: the smallest one a message is cut into (WANTRAIN_TEST_MIN_FRAG: timing only, small test models)
    MIN_FRAG = int(os.environ.get("WANTRAIN_TEST_MIN_FRAG", str(64 << 10)))

    def _wire_multi(self, keys, socks, link: LinkSpec):
        for k in keys:
            self.conns[k] = socks[0]
            # Every stream's socket is closed at the end, not only the first.
            for i, sk in enumerate(socks[1:], 1):
                self.conns[k + (i,)] = sk
        if self._relay:
            self._tied_socks[keys[0]] = len(socks)  # one END per socket (_tied_end)
        q = queue.Queue()  # engine-facing frame queue
        for k in keys:
            self.send_qs[k] = q
        frag_q = queue.Queue()  # encoded fragments, consumed by N workers
        delay = DelayedDelivery(link, self.inbox.put)
        reasm = fr.Reassembler()
        rlock = threading.Lock()
        te = threading.Thread(target=self._mux_encoder, args=(q, frag_q, len(socks)), daemon=True)
        te.start()
        self._send_threads.append(te)
        for sk in socks:
            ts = threading.Thread(
                target=self._frag_sender,
                args=(sk, frag_q, SenderPacer(link), _hop(keys[0])),
                daemon=True,
            )
            ts.start()
            self._send_threads.append(ts)
            self._start_reader(self._reader_shared, (sk, delay, reasm, rlock, keys[0]))

    def _encode_body(self, ftype, body, codec):
        """-> (codec_id, payload, shape).

        The wire carries the exact bytes of the tensor's declared boundary dtype.
        No quantization happens here: narrowing, when the recipe declares any, is
        done inside the model so the single-device baseline runs it too.
        """
        if not isinstance(body, torch.Tensor):
            return fr.OBJ, serial.dumps(body), ()
        if codec == "planes":
            from wantrain.codec.planes import encode
            from wantrain.model import precision

            bdt = precision.spec(self.spec.model.boundary_dtype)["torch_dtype"]
            payload, shape = encode(body.to(bdt))
            return fr.PLANES, payload, shape
        if codec == "fp8_planes":
            bdt = self.spec.model.boundary_dtype
            if bdt.startswith("fp8"):
                # The recipe's fp8 boundary is scaled (q * 2^k per token), because an
                # unscaled cast zeroes activation gradients. A raw-fp8 codec asserts
                # exact e4m3 representability and rightly rejects a scaled tensor, so
                # the scaled wire format carries the exponents alongside the bytes.
                from wantrain.codec.planes import encode_scaled_fp8

                payload, shape = encode_scaled_fp8(body, "e5m2" if bdt.endswith("e5m2") else "e4m3")
                return fr.PLANES, payload, shape
            if ftype == fr.FWD or self.spec.model.fp8_boundaries:
                from wantrain.codec.fp8_planes import encode

                payload, shape = encode(body)
                return fr.FP8_PLANES, payload, shape
            if body.dtype == torch.float32:
                from wantrain.codec.f32_planes import encode

                payload, shape = encode(body)
                return fr.F32_PLANES, payload, shape
        if (
            body.is_floating_point()
            and body.dtype != torch.float32
            and self.spec.model.boundary_dtype == "fp32"
        ):
            # The raw codec carries the declared fp32 boundary. A bf16/fp16 compute tensor
            # widens exactly, and _decode_body narrows it back to the compute dtype, so no bit
            # changes. Without the cast a bf16 recipe died on its first send ("unsupported
            # dtype torch.bfloat16"): found on the first real GPU run, gpt124m bf16, 4 stages.
            body = body.float()
        codec_id, payload, shape = fr.tensor_payload(body)
        return codec_id, payload, shape

    def _decode_body(self, f):
        """Received tensor, promoted to the compute dtype.

        A wire-quantized recipe puts a narrower dtype on the wire than the block
        math runs in; letting that dtype propagate would silently change the
        compute precision downstream, so the promotion is explicit here.

        An arch "hf" cut is HF's residual stream, fp32 whatever the compute
        dtype (ADR-0009 D10), and so is the gradient that crosses it: both are
        used as they arrive. Cast to a bf16 compute dtype they would be
        narrowed, and the run would leave the vanilla forward's bits.
        """
        t = f.tensor()
        if self.hf:
            return t
        from wantrain.model import precision

        want = precision.spec(self.spec.model.compute_dtype)["torch_dtype"]
        return t.to(want) if t.dtype != want and t.is_floating_point() else t

    def _mux_encoder(self, q, frag_q, n_senders=1):
        """Encode frames once, split into fixed fragments, fan out."""
        try:
            self._mux_encoder_loop(q, frag_q, n_senders)
        except Exception as e:  # noqa: BLE001 — see _die
            self._die("mux_encoder", e)

    def _mux_encoder_loop(self, q, frag_q, n_senders):
        while True:
            item = q.get()
            if item is None:
                # Forward the shutdown to every fragment sender, behind any
                # fragments already queued, so each sender drains before it
                # exits (see _finish).
                for _ in range(n_senders):
                    frag_q.put(None)
                return
            ftype, step, mb, chunk, body, codec = item
            if isinstance(body, _RawFrag):
                f = body.frame
                frag_q.put(
                    (ftype, step, mb, chunk, f.payload, f.codec, f.shape, f.frag_idx, f.frag_cnt)
                )
                continue
            t0 = time.monotonic()
            codec_id, payload, shape = self._encode_body(ftype, body, codec)
            size = frag_size(len(payload), n_senders, self.MIN_FRAG, self.MULTI_FRAG)
            frags = [payload[i : i + size] for i in range(0, len(payload), size)] or [b""]
            for i, part in enumerate(frags):
                frag_q.put((ftype, step, mb, chunk, part, codec_id, shape, i, len(frags)))
            # Encode + enqueue only; the wire time is each fragment's "send".
            self._ev(
                "encode", step, mb, chunk, t0, nbytes=len(payload), frags=len(frags), frag_size=size
            )

    def _send_raw(self, sock, pacer, hop, item):
        """A forwarded TIED fragment: its bytes and header fields as received,
        paced and sealed like every frame on this socket (BA-435)."""
        ftype, step, mb, chunk, raw, _ = item
        t0 = time.monotonic()
        f = raw.frame
        pacer.pace(len(f.payload))
        fr.send_frame(
            sock,
            ftype,
            step,
            mb,
            f.payload,
            chunk=chunk,
            codec=f.codec,
            shape=f.shape,
            frag_idx=f.frag_idx,
            frag_cnt=f.frag_cnt,
        )
        self._ev("send", step, mb, chunk, t0, hop=hop, nbytes=len(f.payload), frags=1)

    def _frag_sender(self, sock, frag_q, pacer, hop):
        while True:
            job = frag_q.get()
            if job is None:
                self._send_end(sock)
                return
            ftype, step, mb, chunk, part, codec_id, shape, fi, fc = job
            t0 = time.monotonic()
            pacer.pace(len(part))
            try:
                fr.send_frame(
                    sock,
                    ftype,
                    step,
                    mb,
                    part,
                    chunk=chunk,
                    codec=codec_id,
                    shape=shape,
                    frag_idx=fi,
                    frag_cnt=fc,
                )
            except OSError as e:
                self._die("frag_sender", e)
                return
            self._ev("send", step, mb, chunk, t0, hop=hop, nbytes=len(part), frag=fi)

    def _reader_shared(self, sock, delay, reasm, rlock, src=None):
        try:
            while True:
                f = fr.recv_frame(sock)
                if f.type == fr.END:
                    if self._relay:
                        self._tied_end(src)
                    return
                if self._relay and f.type == fr.TIED and self._tied_seen(f, src):
                    continue
                with rlock:
                    whole = reasm.push(f)
                    spread = reasm.spread_s
                if whole is not None:
                    self._ev_recv(whole, spread, src)
                    delay.push(self._predecode(whole, src), len(whole.payload))
        except _TiedRefused:
            return
        except (ConnectionError, OSError) as e:
            self._die("reader", e)

    def _listen(self):
        lst = socket.socket()
        lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TRANSPORT-I10: congestion control and buffers for long paths; accepted sockets inherit them.
        print(f"WT_NET listen {json.dumps(tuning.tune(lst))}", flush=True)
        # Real-WAN endpoints: neighbours on other machines dial this listener.
        bind_host = "0.0.0.0" if self.spec.endpoints is not None else self.spec.host  # nosec B104
        lst.bind((bind_host, self._listen_port(self.rank, self.peer)))
        lst.listen(16)
        return lst

    def _dial(self, stage, peer):
        # Retries until the listener is up (ranks boot at different speeds over
        # a WAN). dial_timeout_s > 0 bounds it, so a dead neighbour surfaces as
        # an error instead of a process that waits forever.
        limit = self.spec.dial_timeout_s
        deadline = time.monotonic() + limit if limit and limit > 0 else None
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise ConnectionError(
                    f"could not reach stage {stage} peer {peer} at "
                    f"{self._host(stage, peer)}:{self._port(stage, peer)} "
                    f"within {limit:.0f}s"
                )
            c = None
            try:
                # TRANSPORT-I10: tuned before the handshake, which fixes the window scale.
                c, applied = tuning.connect(
                    self._host(stage, peer), self._port(stage, peer), timeout=10
                )
                if not self._net_logged:
                    self._net_logged = True
                    print(f"WT_NET dial {json.dumps(applied)}", flush=True)
                # clear the connect-timeout: data transfers over WAN must
                # block indefinitely, not time out mid-send on large tensors
                c.settimeout(None)
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return c
            except OSError:
                if c is not None:  # reset right after connect: drop it and redial
                    c.close()
                time.sleep(0.05)

    def _wire(self, key, sock, link: LinkSpec):
        """Register a connection: sender thread + reader thread with shim.
        Under wire "native" an intra connection's queue serves need-first
        (RUNTIME-I13, BA-14)."""
        self.conns[key] = sock
        if key[0] == "intra" and (self.native or self.pieces_mode):
            q = xc.NeedFirstQueue(fifo=self._fifo)
        else:
            q = queue.Queue()
        self.send_qs[key] = q
        pacer = SenderPacer(link)
        delay = DelayedDelivery(link, self.inbox.put)
        self._start_reader(self._reader, (sock, delay, key))
        t = threading.Thread(target=self._sender, args=(sock, q, pacer, _hop(key)), daemon=True)
        t.start()
        self._send_threads.append(t)

    def _start_reader(self, target, args):
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        self._read_threads.append(t)

    def _send_end(self, sock):
        """Graceful end of this direction: END, then half-close. The peer's
        reader returns on END, so a later EOF is never mistaken for a crash,
        and our reader keeps draining until the peer says the same."""
        try:
            fr.send_end(sock)
            sock.shutdown(socket.SHUT_WR)
        except OSError as e:
            self._die("sender", e)

    def _connect_mesh(self):
        """Wide topology: each slot dials every peer of the next stage and its
        lower intra peers, and accepts the previous stage's peers and its
        higher intra peers. Every socket is authenticated against the pinned
        slot table and sealed before any frame; an accepted socket names its
        slot in a sealed HELLO, checked against the slot it proved (RUNTIME-I16)."""
        s = self.spec
        lst = self._listen()
        # outbound: all peers of the next stage, then intra to lower peers
        dials = []
        if self.rank < self.S - 1:
            for q in range(self.widths[self.rank + 1]):
                dials.append(
                    (
                        ("next", q),
                        self.rank + 1,
                        q,
                        self._link(s.links[self.rank], self.rank + 1, q),
                    )
                )
        for p in range(self.peer):
            dials.append((("intra", p), self.rank, p, self._link(None, self.rank, p)))
        results = [
            (key, _Result(lambda st=st, pe=pe: self._dial_secure(st, pe, hello=True)), st, pe, link)
            for key, st, pe, link in dials
        ]
        # inbound: every peer of the previous stage + higher intra peers
        expected = []
        if self.rank > 0:
            expected += [(self.rank - 1, q) for q in range(self.widths[self.rank - 1])]
        expected += [(self.rank, p) for p in range(self.peer + 1, self.w)]
        if s.slot_peers is not None:
            ids = {self._slot_id(st, pe) for st, pe in expected}
        elif s.trusted_peers is not None:
            ids = set(s.trusted_peers)
        else:
            ids = None

        def admit(sk):
            auth = self._auth_accepted(sk)
            pid = auth["peer_id"]
            # An id no expected slot holds is refused before it may send
            # anything: no frame of an unknown key is ever parsed.
            if ids is not None and pid not in ids:
                raise _Refused("peer", peer=pid)
            ss = self._seal(sk, auth)
            try:
                hello = fr.recv_frame(ss)
                claim = hello.obj() if hello.type == HELLO else None
            except TimeoutError:
                raise _Refused("silent", peer=pid) from None
            except Exception:  # noqa: BLE001 -- any malformed HELLO is a refusal
                raise _Refused("hello", peer=pid) from None
            if (
                not isinstance(claim, (tuple, list))
                or len(claim) != 2
                or not all(type(x) is int for x in claim)
            ):
                raise _Refused("hello", peer=pid)
            st, pe = claim
            if st not in (self.rank - 1, self.rank) or st < 0:
                raise _Refused("stage", st, pe, pid)
            if not 0 <= pe < self.widths[st] or (st == self.rank and pe <= self.peer):
                raise _Refused("range", st, pe, pid)
            if not self._admissible(pid, self._slot_id(st, pe)):
                raise _Refused("peer", st, pe, pid)
            return (st, pe), (ss, auth), st, pe, pid

        inbound = self._accept_until_bound(
            lst,
            expected,
            admit,
            [r for _, r, _, _, _ in results],
            lambda k: f"intra{k[1]}" if k[0] == self.rank else f"prev{k[1]}",
        )
        for key, res, st, pe, link in results:
            ss, auth = res.get()
            self._bound(_hop(key), ss, auth, st, pe)
            self._wire(key, ss, link)
        for (st, pe), (ss, _) in sorted(inbound.items()):
            if st == self.rank:
                self._wire(("intra", pe), ss, self._link(None, st, pe))
            else:
                self._wire(("prev", pe), ss, self._link(s.links[st], st, pe))

    def _link(self, base, st, pe) -> LinkSpec:
        """The LinkSpec of this slot's connection to slot (st, pe): the link's
        emulation (`base`, a LinkCfg dict; None for intra: no delay, or
        WANTRAIN_TEST_INTRA_LATENCY_MS's, timing only), plus under pacing
        "shared" this peer's egress and ingress budgets and, between two
        sites, the uplink budget of this direction (BA-16, BA-55): they pace
        the frame bytes handed to the sealed socket, where SenderPacer does."""
        link = LinkSpec(**base) if base else LinkSpec(latency_ms=self._intra_ms)
        s = self.spec
        if s.pacing != "shared":
            return link
        if not self._budgets:
            paths = s.budget_paths()
            mbps = {("egress", self.rank, self.peer): s.peer_egress_mbps[self.rank][self.peer]}
            mbps[("ingress", self.rank, self.peer)] = s.peer_ingress_mbps[self.rank][self.peer]
            for kind, a, b in paths:
                if kind == "uplink" and a == s.sites[self.rank][self.peer]:
                    mbps[(kind, a, b)] = s.uplink_mbps[f"{a}->{b}"]
            self._budgets = {k: SharedBudget(float(v), paths[k]) for k, v in mbps.items()}
        link.egress = self._budgets[("egress", self.rank, self.peer)]
        link.ingress = self._budgets[("ingress", self.rank, self.peer)]
        here, there = s.sites[self.rank][self.peer], s.sites[st][pe]
        if here != there:
            link.uplink = self._budgets[("uplink", here, there)]
        return link

    def _die(self, where, exc):
        """A wire thread lost its socket outside a clean shutdown. Threads must
        never swallow this: a reader that returns silently leaves the main loop
        blocked in inbox.get() for the rest of the run. The failure is pushed
        into the inbox, the one channel the main loop is guaranteed to drain."""
        if self._closing:
            return
        self.inbox.put(("fatal", f"{where}: peer connection lost ({type(exc).__name__}: {exc})"))

    def _predecode(self, f, src=None):
        """Decode boundary tensors on the reader thread, not the main loop:
        inflating a multi-MB boundary is milliseconds of CPU that would land
        inside the compute window the overlap condition assumes is free. The
        reader is idle between frames and zlib/zstd release the GIL. GRADS and
        PARAMS objects are decoded here too, never promoted (BA-15); a body
        that does not decode is handed to the main loop as malformed. `src`
        is the connection the frame arrived on (its authenticated slot)."""
        if f.type in (fr.FWD, fr.BWD):
            return ("net_dec", f, self._decode_body(f), src)
        if f.type in (fr.GRADS, fr.PARAMS):
            try:
                return ("net_obj", f, f.obj(), src)
            except (ValueError, ConnectionError) as e:
                return ("net_bad", f, e, src)
        return ("net", f, None, src)

    def _reader(self, sock, delay, src=None):
        reasm = fr.Reassembler()
        try:
            while True:
                f = fr.recv_frame(sock)
                if f.type == fr.END:
                    if self._relay:
                        self._tied_end(src)
                    return
                # A stage between the relay's ends forwards each TIED fragment
                # from here, never waiting for its own compute (RUNTIME-I19).
                if self._relay and f.type == fr.TIED and self._tied_seen(f, src):
                    continue
                whole = reasm.push(f)
                if whole is not None:
                    self._ev_recv(whole, reasm.spread_s, src)
                    delay.push(self._predecode(whole, src), len(whole.payload))
        except _TiedRefused:
            return
        except (ConnectionError, OSError) as e:
            self._die("reader", e)

    def _sender(self, sock, q, pacer, hop):
        """Sender thread with adaptive microbatching: when a frame's transfer
        time escapes the blind-window budget, halve the fragment granularity;
        recover exponentially after clean sends."""
        try:
            self._sender_loop(sock, q, pacer, hop)
        except Exception as e:  # noqa: BLE001 — any death here strands the run
            self._die("sender", e)

    def _sender_loop(self, sock, q, pacer, hop):
        budget = self.spec.adapt_budget_ms / 1e3
        frag_size = None
        clean = 0
        stamped = isinstance(q, xc.NeedFirstQueue)
        while True:
            if stamped:
                item, tq, tp = q.pop_with_times()
            else:
                item = q.get()
            if item is None:
                self._send_end(sock)
                return
            ftype, step, mb, chunk, body, codec = item
            t0 = time.monotonic()
            if isinstance(body, _RawFrag):
                self._send_raw(sock, pacer, hop, item)
                continue
            xb = body if isinstance(body, _Exchange) else None
            extra = {}
            if xb is not None:
                # An exchange chunk: its D2H copy ran on the copy engine; the
                # body is an OBJ, the exact byte image (BA-56), never a tensor
                # body a wire codec could narrow.
                tensors = xb.fut.result()
                body = ("x", xb.env, tensors)
                extra = dict(ft=ftype, tb=xb.nbytes, key=list(xb.key), tq=tq, tp=tp)
            codec_id, payload, shape = self._encode_body(ftype, body, codec)
            # an empty payload (a TIED frame of no rows) is one empty fragment
            size = frag_size or max(1, len(payload))
            frags = [payload[i : i + size] for i in range(0, len(payload), size)] or [b""]
            for i, part in enumerate(frags):
                pacer.pace(len(part))
                fr.send_frame(
                    sock,
                    ftype,
                    step,
                    mb,
                    part,
                    chunk=chunk,
                    codec=codec_id,
                    shape=shape,
                    frag_idx=i,
                    frag_cnt=len(frags),
                )
            dt = time.monotonic() - t0
            if xb is not None:
                self.holds.sub(xb.hold, xb.nbytes)
            self._ev(
                "send",
                step,
                mb,
                chunk,
                t0,
                hop=hop,
                nbytes=len(payload),
                frags=len(frags),
                frag_size=size,
                **extra,
            )
            if budget > 0:
                per_frag = dt / len(frags)
                if per_frag > budget and size > self.spec.frag_min_bytes:
                    frag_size = max(self.spec.frag_min_bytes, size // 2)
                    clean = 0
                elif per_frag < budget / 4:
                    clean += 1
                    if clean >= 8 and frag_size is not None:
                        frag_size = min(len(payload), frag_size * 2)
                        if frag_size >= len(payload):
                            frag_size = None
                        clean = 0

    # ---------------- helpers ----------------

    def _take_events(self, upto=None):
        """Every event recorded so far, exactly once: the list is swapped
        under the lock, so an append racing the take lands in the next one.
        With `upto` (shard "pieces", BA-45), op and stall events of later
        steps stay queued for their own step's summary."""
        with self._tel_lock:
            if upto is None:
                events, self.telemetry = self.telemetry, []
            else:
                events, keep = [], []
                for e in self.telemetry:
                    later = e.get("ev") in _STEP_EVENTS and e.get("step", -1) > upto
                    (keep if later else events).append(e)
                self.telemetry = keep
        return events

    def _append_telemetry(self, events):
        """Append to the slot's event log, so long runs expose live busy/stall
        metrics and survive crashes with telemetry intact."""
        if not events:
            return
        os.makedirs(self.spec.out_dir, exist_ok=True)
        path = os.path.join(
            self.spec.out_dir, f"{self.spec.tag}_rank{self.rank}p{self.peer}_telemetry.jsonl"
        )
        with open(path, "a") as f:
            f.write("".join(json.dumps(ev) + "\n" for ev in events))

    def _flush_telemetry(self):
        self._append_telemetry(self._take_events())

    def _emit_step_telemetry(self, step, step_ms, upto=None):
        """RUNTIME-I9: one WT_TELEM line per optimizer step, from monotonic
        stamps only (no tensor op, no device sync, no RNG). The block's own
        duration is reported as the next summary's emit_us. With `upto`
        (shard "pieces") the log also records when the summary took its
        events (a "summary" marker)."""
        t0 = time.perf_counter()
        events = self._take_events(upto)
        if upto is not None:
            now = time.monotonic()
            events.append(dict(ev="summary", step=step, mb=-1, vs=-1, t0=now, t1=now))
        self._append_telemetry(events)
        summary = summarize_step(events, step, self.rank, self.peer, step_ms, self._emit_us)
        line = "WT_TELEM " + json.dumps(summary)
        if len(line.encode()) <= TELEM_LINE_MAX:
            print(line, flush=True)
        else:
            print(
                f"WT_WARN telemetry summary of step {step} is {len(line.encode())} bytes, "
                f"above {TELEM_LINE_MAX}; not sent",
                flush=True,
            )
        self._emit_us = (time.perf_counter() - t0) * 1e6

    def _stream_loss(self, step, mb, value):
        """Live loss feed for long runs: one JSONL line per microbatch, so
        training is observable (and killable) without waiting for _finish."""
        path = os.path.join(self.spec.out_dir, f"{self.spec.tag}_loss_stream.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps({"step": step, "mb": mb, "loss": value, "t": time.time()}) + "\n")

    def _ckpt_path(self):
        return os.path.join(
            self.spec.out_dir, f"{self.spec.tag}_ckpt_rank{self.rank}p{self.peer}.pt"
        )

    def _checkpoint(self, step):
        # The loss history rides along, so a resumed run still reports the
        # whole sequence and can be diffed bit for bit against an
        # uninterrupted one.
        if self.pieces_mode:
            payload = self._pieces_payload(step)
        else:
            payload = {
                "step": step,
                "chunks": {vs: self._chunk_state(vs) for vs in self.my_vs},
                "opt": self._opt_state(),
                # Part of the state the byte-exact rule compares, and restored
                # on resume so a healed run continues the same RNG stream.
                "rng": torch.get_rng_state(),
                "losses": {f"{s}:{m}": v for (s, m), v in self.losses.items() if s <= step},
                # What a resume must match before it applies any of the above.
                "pins": self.spec.pins(),
            }
            if self.moe:
                payload["moe_aux"] = self._moe_aux_upto(step)
        # RUNTIME-I17: an immutable wave (state.pt, then wave.json, then the
        # directory renamed into place), and only then the alias (_ckpt_path),
        # derived from it in seqlock order (checkpoint.save_shard), then
        # WT_WAVE. Only the serialization runs here, while the payload still
        # references the live tensors; the fsync, the sha256 of wave.json,
        # the renames, the alias and the pruning run on one background commit,
        # off the step (HANDOUT-M2 "I/O per wave", THREAT_MODEL §3). A second
        # commit of a step writes nothing, so the last step, checkpointed
        # periodically and finally, is announced once.
        self._await_wave()
        final = wave_dir(self.spec.out_dir, self.spec.tag, self.rank, self.peer, step)
        staged = None
        if os.path.lexists(os.path.normpath(final)):
            save_fn = _held_step
        else:
            os.makedirs(self._slot_dir(), exist_ok=True)
            staged = os.path.join(self._slot_dir(), f"{_STAGED_PREFIX}{step:08d}.pt")
            torch.save(payload, staged)

            def save_fn(tmp, staged=staged):
                os.rename(staged, tmp)

        self._wave_commit = _Result(lambda: self._commit_wave(step, save_fn, staged))

    def _chunk_state(self, vs) -> dict:
        """The chunk's state_dict; stage 0's working copy of a split tied table
        is not saved as the table (BA-417): the Head's checkpoint holds it."""
        sd = self.chunks[vs].state_dict()
        for k in self._tied_keys:
            sd.pop(k, None)
        return sd

    def _moe_aux_upto(self, step) -> list:
        return [list(r) for r in self.moe_aux if r[0] <= step]

    def _opt_state(self) -> dict:
        """The optimizer state a whole-tensor checkpoint records. An owner-less
        ZeRO-1 slot records no tensor's state under its peers' hyperparameters,
        so the wave and the manifest (validation/state.py) stay well-defined."""
        if self.opt is not None:
            return self.opt.state_dict()
        group = optim.hparams(self.spec.lr, self.spec.optimizer)
        return {"state": {}, "param_groups": [{**group, "params": []}]}

    def _load_opt(self, opt_state, where):
        """Load a checkpoint's optimizer state; an owner-less slot only checks
        that the state names no tensor (it cannot own one)."""
        if self.opt is not None:
            self.opt.load_state_dict(opt_state)
            return
        ok = isinstance(opt_state, dict) and not opt_state.get("state")
        groups = opt_state.get("param_groups") if ok else None
        if not groups or any(g.get("params") for g in groups):
            raise RuntimeError(
                f"resume refused: {where} holds optimizer state for slot "
                f"r{self.rank}p{self.peer}, which owns no trainable tensor"
            )

    def _nontrainable(self) -> dict:
        """The stage's buffers and frozen parameters, by chunk name."""
        chunk = self.chunks[self.my_vs[0]]
        out = dict(chunk.named_buffers())
        out.update({n: p for n, p in chunk.named_parameters() if not p.requires_grad})
        return out

    def _pieces_payload(self, step) -> dict:
        """RUNTIME-I12, BA-18: one slot file in logical layout (logical_ckpt):
        this peer's pieces (master, m, v, step), the hyperparameters, the RNG,
        the loss history, the non-trainable replica, the working-copy digests,
        the pins and the embedded manifest naming step, width, peer and the
        full layout."""
        st = self.popt.state
        mine = {}
        for n in self.pnames:
            s = st[self.master[n]]
            mine[n] = {
                "master": self.master[n],
                "exp_avg": s["exp_avg"],
                "exp_avg_sq": s["exp_avg_sq"],
                "step": s["step"],
            }
        out = {
            "step": step,
            "pieces": mine,
            "hparams": {k: v for k, v in self.popt.param_groups[0].items() if k != "params"},
            "rng": torch.get_rng_state(),
            "losses": {f"{s}:{m}": v for (s, m), v in self.losses.items() if s <= step},
            "nontrainable": self._nontrainable(),
            "working": {n: lc.digest(self.pby[n].data) for n in self.pnames},
            "pins": self.spec.pins(),
            "pieces_manifest": lc.build_manifest(
                step,
                self.w,
                self.peer,
                {n: self.layout[n] for n in self.pnames},
                mine,
                {n: self.shapes[n] for n in self.pnames},
            ),
        }
        if self.moe:
            out["moe_aux"] = self._moe_aux_upto(step)
        return out

    def _commit_wave(self, step, save_fn, staged):
        # arch "hf": every wave lists the run's buffer record (RUNTIME-I19, I-15,
        # BA-460); a dense slot's call is today's.
        kw = {"extra_files": {HF_BUFFERS: hf_buffers_bytes(self.spec.model)}} if self.hf else {}
        try:
            _, committed = commit_wave(
                self.spec.out_dir, self.spec.tag, self.rank, self.peer, step, save_fn, **kw
            )
        finally:
            if staged is not None and os.path.lexists(staged):
                os.unlink(staged)
        if committed:
            print(f"WT_WAVE {step}", flush=True)

    def _await_wave(self):
        """Wait for the checkpoint commit in flight, if any; its failure is
        this stage's failure. One commit at a time: a checkpoint waits for
        the previous one, and the run waits for the last before it ends."""
        pending, self._wave_commit = self._wave_commit, None
        if pending is not None:
            pending.get()

    def _settle_wave(self):
        """On a failing run, let the commit in flight finish before the error
        ends the process, whose exit stops the commit's daemon thread: a step
        checkpointed before the failure stays a committed, announced wave the
        agent's exit scan ships, as when the commit ran on the step. The
        commit's own failure is logged; the run's error is the one raised."""
        try:
            self._await_wave()
        except Exception as e:  # noqa: BLE001 -- the run's error propagates
            print(
                f"WT_WAVE_FAILED rank={self.rank} peer={self.peer} {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )

    def _slot_dir(self):
        return os.path.dirname(
            os.path.normpath(wave_dir(self.spec.out_dir, self.spec.tag, self.rank, self.peer, 0))
        )

    def _drop_abandoned_waves(self, start_step):
        """BA-6, before the first step: a resume from step n drops the slot's
        waves newer than n (an abandoned timeline), a launch that does not
        resume (start_step 0) drops every wave of the slot under its tag, so
        no later commit of one of those steps finds a stale wave held."""
        self._sweep_wave_temps()
        dropped = drop_waves_after(
            self.spec.out_dir, self.spec.tag, self.rank, self.peer, start_step - 1
        )
        if dropped:
            print(
                f"WT_DROPPED_WAVES rank={self.rank} peer={self.peer} "
                f"after={start_step - 1} steps={dropped}",
                flush=True,
            )

    def _sweep_wave_temps(self):
        """Remove the private temp directories a killed writer of this slot
        left (commit_wave's .tmp-step-*, a wave drop's .drop-*, a staged
        state.pt .staged-step-*): the agent
        stops a stage with SIGTERM/SIGKILL, so no cleanup runs, and each kill
        during a checkpoint would otherwise keep a whole state.pt on disk for
        good. Safe before the first step: this stage is the slot's only
        writer, and nothing reads a temp."""
        slot = self._slot_dir()
        try:
            names = sorted(
                n
                for n in os.listdir(slot)
                if n.startswith((".tmp-step-", ".drop-", _STAGED_PREFIX))
            )
        except FileNotFoundError:
            return
        for n in names:
            p = os.path.join(slot, n)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.unlink(p)
        if names:
            print(
                f"WT_DROPPED_WAVE_TEMPS rank={self.rank} peer={self.peer} names={names}",
                flush=True,
            )

    def _resume(self):
        """Load this slot's wave of spec.resume_step (its newest wave when
        resume_step is None, local mode) and return the step to start from.

        RUNTIME-I17: only the wave is read, never the alias. Fatal, before any
        state is applied, when the wave or its wave.json is missing, its
        wave.json names another step, a listed file is missing or has another
        sha256 or size, a file is unlisted, or the recipe pins differ: the
        spec said resume, so every peer is resuming, and a slot that silently
        restarted at step 0 (or from another step) would leave the pipeline
        waiting on step-tagged frames that are never coming.
        """
        if self.pieces_mode:
            return self._resume_logical()
        spec = self.spec
        step = spec.resume_step
        if step is None:
            held = slot_waves(spec.out_dir, spec.tag, self.rank, self.peer)
            if not held:
                raise RuntimeError(
                    f"spec.resume is set but this box holds no wave of slot "
                    f"r{self.rank}p{self.peer} under tag {spec.tag!r}. Every slot must "
                    f"hold its wave before a resume launch."
                )
            step = held[-1][0]
        d = wave_dir(spec.out_dir, spec.tag, self.rank, self.peer, step)
        if not os.path.isdir(d):
            raise RuntimeError(
                f"spec.resume is set but wave {d} (step {step}) does not exist on this box. "
                f"Every slot must hold its wave before a resume launch."
            )
        try:
            validate_wave(d, step)
        except ValueError as e:
            raise RuntimeError(f"resume refused: {e}") from e
        path = os.path.join(d, WAVE_STATE)
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Before any state is applied: the payload must be the step its
        # wave.json names, or this slot would start at a step the rest of the
        # fleet is not at.
        if not isinstance(state, dict):
            raise RuntimeError(
                f"resume refused: {path} holds a {type(state).__name__}, not a checkpoint"
            )
        if state.get("step") != step:
            raise RuntimeError(
                f"resume refused: {path} holds step {state.get('step')!r}, "
                f"its wave names step {step}"
            )
        if "pieces" in state:
            # A logical-layout wave read by shard "zero1": names[k::w] of the
            # logical tensors (RUNTIME-I12), so the shard rolls back.
            self._wave_cache = {(self.peer, step): (path, state)}
            logical = self._read_logical(step)
            self._apply_logical_zero1(logical)
            return self._resumed(step)
        self._check_resume_pins(path, state)
        if self.hf:
            check_hf_buffers(d, self.spec.model.hf_buffers)
        for vs in self.my_vs:
            sd = dict(state["chunks"][vs])
            if self._tied_keys:
                # stage 0's working copy is no state of this slot: kept as built
                own = self.chunks[vs].state_dict()
                sd.update({k: own[k] for k in self._tied_keys})
            self.chunks[vs].load_state_dict(sd)
        self._load_opt(state["opt"], path)
        if state.get("rng") is not None:
            torch.set_rng_state(state["rng"])
        for k, v in (state.get("losses") or {}).items():
            s_, m_ = k.split(":")
            self.losses[(int(s_), int(m_))] = float(v)
        if self.moe:
            self.moe_aux = [list(r) for r in state.get("moe_aux") or [] if r[0] <= step]
        start = int(state["step"]) + 1
        print(
            f"WT_RESUME rank={self.rank} start_step={start} (loaded wave step-{step:08d})",
            flush=True,
        )
        return start

    def _check_resume_pins(self, path, state, logical=False):
        """Refuse, before any state is applied, a shard written under another
        recipe (RUNTIME-I15): its bytes would continue a run this spec does
        not describe. Ownership, schedule, addresses and devices of the same
        device type may change. BA-53: "zero1_widths" is skipped only for a
        pieces payload (width-free, checked through its manifest) and for a
        zero1 wave a shard "pieces" resume reads logically (BA-57); a zero1
        shard is never read as a shard at another width."""
        stored = state.get("pins")
        if stored is None:
            raise RuntimeError(f"{path} records no recipe pins; refusing to resume from it")
        want = self.spec.pins()
        if self.hf:
            slot = device_class_change(want.get("device_classes"), stored.get("device_classes"))
            if slot is not None:
                raise RuntimeError(
                    f"{path} was written under another device class map: slot {slot} moved "
                    f"class (RUNTIME-I15); refusing to resume from it"
                )
        if ("pieces" in state and "pieces_manifest" in state) or (logical and self.pieces_mode):
            want.pop("zero1_widths")
            stored = {k: v for k, v in stored.items() if k != "zero1_widths"}
        diff = first_pin_difference(want, stored)
        if diff is not None:
            raise RuntimeError(
                f"{path} was written under another recipe: pin {diff!r} differs; "
                f"refusing to resume from it"
            )
        step = int(state["step"])
        if self.spec.steps < step + 1:
            raise RuntimeError(
                f"steps={self.spec.steps} is below the first step a resume from {path} "
                f"would run ({step + 1}); refusing to resume from it"
            )

    # ---------------- logical resume (RUNTIME-I12, BA-19, BA-51, BA-57) ----------------

    def _read_slot(self, k, step):
        """Old slot (rank, k)'s wave at `step`, validated (checkpoint.
        validate_wave) and loaded once: -> (wave dir, payload)."""
        cache = self._wave_cache
        if (k, step) in cache:
            return cache[(k, step)]
        spec = self.spec
        d = wave_dir(spec.out_dir, spec.tag, self.rank, k, step)
        if not os.path.isdir(d):
            raise RuntimeError(f"resume refused: slot p{k} has no wave at step {step} ({d})")
        try:
            validate_wave(d, step)
        except ValueError as e:
            raise RuntimeError(f"resume refused: {e}") from e
        state = torch.load(os.path.join(d, WAVE_STATE), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("step") != step:
            got = state.get("step") if isinstance(state, dict) else type(state).__name__
            raise RuntimeError(
                f"resume refused: {d} holds step {got!r}, its wave names step {step}"
            )
        cache[(k, step)] = (d, state)
        return d, state

    def _peers_with_wave(self, step):
        """Peer indices of this stage with a wave directory at `step`."""
        spec = self.spec
        root = os.path.join(spec.out_dir, "waves", spec.tag)
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            return []
        out = []
        for n in names:
            m = re.fullmatch(r"r(\d+)p(\d+)", n)
            if m and int(m.group(1)) == self.rank:
                k = int(m.group(2))
                if os.path.isdir(wave_dir(spec.out_dir, spec.tag, self.rank, k, step)):
                    out.append(k)
        return sorted(out)

    def _wave_width(self, state):
        if "pieces" in state:
            man = state.get("pieces_manifest")
            if not isinstance(man, dict) or type(man.get("width")) is not int:
                raise RuntimeError("resume refused: a pieces payload without its manifest")
            return man["width"]
        try:
            return int(state["pins"]["zero1_widths"][self.rank])
        except (KeyError, TypeError, IndexError):
            raise RuntimeError(
                "resume refused: a zero1 payload without its zero1_widths pin"
            ) from None

    def _newest_complete_wave(self):
        """BA-51: the newest step at which old slot p0's wave validates and
        every slot p0..p(w_old-1) it names validates too; never this slot's
        own waves (a raised width resumes slots that hold none)."""
        spec = self.spec
        for step, _ in reversed(slot_waves(spec.out_dir, spec.tag, self.rank, 0)):
            try:
                _, p0 = self._read_slot(0, step)
                for k in range(1, self._wave_width(p0)):
                    self._read_slot(k, step)
            except RuntimeError:
                continue
            return step
        raise RuntimeError(
            f"spec.resume is set but slot r{self.rank}p0 holds no complete wave under tag "
            f"{spec.tag!r}; every slot of the wave must hold its wave before a resume launch"
        )

    def _read_logical(self, step):
        """The stage's logical state at `step` from the old wave's slots, every
        refusal named before any state is applied: a pieces wave through its
        embedded manifests (select_wave, assemble), a whole-tensor zero1 wave
        as names[k::w_old] (from_zero1, BA-57)."""
        _, p0 = self._read_slot(0, step)
        mbs = self.M if self.hosts_final else None
        names = [n for n, p in self.chunks[self.my_vs[0]].named_parameters() if p.requires_grad]
        try:
            if "pieces" in p0:

                def read_manifest(k):
                    d, p = self._read_slot(k, step)
                    if "pieces" not in p:
                        raise ValueError(
                            f"{d}: a wave mixing kinds (slot p{k} is zero1, p0 pieces)"
                        )
                    if "pieces_manifest" not in p:
                        raise ValueError(f"{d}: pieces payload without its manifest")
                    return lc.check_manifest(p["pieces_manifest"], p, d)

                w_old, _ = lc.select_wave(read_manifest, self._peers_with_wave(step))
                payloads = [self._read_slot(k, step) for k in range(w_old)]
                for d, p in payloads:
                    self._check_resume_pins(d, p, logical=True)
                    if self.hf:
                        check_hf_buffers(d, self.spec.model.hf_buffers)
                logical = lc.assemble(payloads, microbatches=mbs)
            else:
                w_old = self._wave_width(p0)
                payloads = [self._read_slot(k, step) for k in range(w_old)]
                for d, p in payloads:
                    if "pieces" in p:
                        raise ValueError(f"{d}: a wave mixing kinds (slot p0 is zero1)")
                    self._check_resume_pins(d, p, logical=True)
                    if self.hf:
                        check_hf_buffers(d, self.spec.model.hf_buffers)
                logical = lc.from_zero1(payloads, names, w_old, microbatches=mbs)
        except ValueError as e:
            raise RuntimeError(f"resume refused: {_name_slot(str(e), self, step)}") from e
        if list(logical["tensors"]) != names:
            raise RuntimeError(
                f"resume refused: the wave holds tensors {list(logical['tensors'])}, this stage {names}"
            )
        mine = self._nontrainable()
        missing = [n for n in mine if n not in logical["nontrainable"]]
        if missing:
            raise RuntimeError(f"resume refused: the wave holds no non-trainable state {missing}")
        return logical

    def _resume_logical(self):
        """Shard "pieces": every slot of the stage resumes from the same step
        (spec.resume_step, else old slot p0's newest complete wave) and the
        old wave's logical state, at any width or split."""
        self._wave_cache: dict = {}
        step = self.spec.resume_step
        if step is None:
            step = self._newest_complete_wave()
        logical = self._read_logical(step)
        own = lc.owned_pieces(logical, {n: self.layout[n] for n in self.pnames}, self.peer)
        with torch.no_grad():
            for n in self.pnames:
                t = logical["tensors"][n]
                if tuple(t["master"].shape) != self.shapes[n]:
                    raise RuntimeError(f"resume refused: {n} has shape {tuple(t['master'].shape)}")
            for n in self.pnames:
                m = self.master[n]
                m.copy_(own[n]["master"])
                self.pby[n].data.copy_(logical["tensors"][n]["master"].to(self.gdt[n]))
                self.popt.state[m] = {
                    "step": own[n]["step"],
                    "exp_avg": own[n]["exp_avg"].to(m.device),
                    "exp_avg_sq": own[n]["exp_avg_sq"].to(m.device),
                }
        self._apply_common(logical)
        with torch.no_grad():
            # MoE selection biases: the replica the wave holds, cut to this
            # peer's master piece (RUNTIME-I18, BA-455)
            for n in self._bias_names:
                self.master[n].copy_(pc.piece(self.pby[n].data, self.layout[n], self.peer))
        self._wave_cache = {}
        return self._resumed(step)

    def _apply_logical_zero1(self, logical):
        names = [n for n, p in self.chunks[self.my_vs[0]].named_parameters() if p.requires_grad]
        w = self.w if self.wide and self.w > 1 else 1
        shard = lc.as_zero1(logical, names, w, self.peer)
        group = {k: v for k, v in logical["hparams"].items() if k != "params"}
        opt_state = {
            "state": {j: st for j, (_, _, st) in enumerate(shard)},
            "param_groups": [{**group, "params": list(range(len(shard)))}],
        }
        by_name = dict(self.chunks[self.my_vs[0]].named_parameters())
        with torch.no_grad():
            for n in names:
                by_name[n].copy_(logical["tensors"][n]["master"])
        self._load_opt(opt_state, "the logical wave")
        self._apply_common(logical)
        self._wave_cache = {}

    def _apply_common(self, logical):
        with torch.no_grad():
            for n, t in self._nontrainable().items():
                t.copy_(logical["nontrainable"][n])
        if logical.get("rng") is not None:
            torch.set_rng_state(logical["rng"])
        for k, v in (logical.get("losses") or {}).items():
            s_, m_ = k.split(":")
            self.losses[(int(s_), int(m_))] = float(v)
        if self.moe:
            # each (step, mb, layer) is kept by the slot owning mb now, so the
            # stage's slots hold every record once (RUNTIME-I18)
            mine = set(self.my_mbs)
            self.moe_aux = [list(r) for r in logical.get("moe_aux") or [] if r[1] in mine]

    def _resumed(self, step):
        start = step + 1
        print(
            f"WT_RESUME rank={self.rank} start_step={start} (loaded wave step-{step:08d})",
            flush=True,
        )
        return start

    def _ev(self, ev, step, mb, vs, t0, **kw):
        e = dict(ev=ev, step=step, mb=mb, vs=vs, t0=t0, t1=time.monotonic(), **kw)
        with self._tel_lock:
            self.telemetry.append(e)

    def _ev_recv(self, f, spread_s, src):
        """A boundary (activation or gradient) became whole on this stage: t0 its first fragment's arrival, t1 its
        last's. Read beside the stalls it shows which wait was the wire's and which the sender's."""
        if f.type in (fr.FWD, fr.BWD) and src is not None:
            t1 = time.monotonic()
            e = dict(ev="recv", step=f.step, mb=f.mb, vs=f.chunk, t0=t1 - spread_s, t1=t1)
            e.update(
                hop=_hop(src), nbytes=len(f.payload), kind="fwd" if f.type == fr.FWD else "bwd"
            )
            with self._tel_lock:
                self.telemetry.append(e)

    def _sim_slow_wait(self, step, t0):
        """WANTRAIN_SIM_SLOW: stretch this op to factor x its compute time,
        inside its timed span, after the compute (no data path is touched)."""
        if self._sim_slow is not None and step >= self._sim_slow[1]:
            time.sleep((self._sim_slow[0] - 1.0) * (time.monotonic() - t0))

    def _route_fwd(self, step, mb, vs_next, x):
        if self.wide:
            # to the downstream peer that owns this microbatch; the chunk field
            # carries the source peer so its input gradient can route back
            q = self.spec.mb_owner[self.rank + 1][mb]
            self.send_qs[("next", q)].put((fr.FWD, step, mb, self.peer, x, self.spec.wire_codec))
        else:
            self.send_qs[("next", 0)].put((fr.FWD, step, mb, 0, x, self.spec.wire_codec))

    def _route_bwd(self, step, mb, vs_prev, g):
        if self.wide:
            q = self.fwd_origin.pop((step, mb), None)
            if q is None:
                q = self.spec.mb_owner[self.rank - 1][mb]
            self.send_qs[("prev", q)].put((fr.BWD, step, mb, self.peer, g, self.spec.wire_codec))
        else:
            self.send_qs[("prev", 0)].put((fr.BWD, step, mb, 0, g, self.spec.wire_codec))

    def _chunk_params(self, vs):
        # Trainable only, in the same traversal order self.params was built in:
        # torch.autograd.grad raises on a tensor that does not require grad, and
        # _store_grads zips the returned gradients against self.params by index,
        # so a filter that disagreed with that one would pair gradients wrongly.
        # Stage 0's working copy of a split tied table is no trainable
        # parameter of this slot (BA-411): the Head owns it.
        wc = self._tied if self._tied_role == "embed" else None
        return [p for p in self.chunks[vs].parameters() if p.requires_grad and p is not wc]

    # ---------------- compute ----------------

    def _compute_ctx(self):
        return compute_autocast(self.spec.model, self.device.type, self._user_autocast)

    def _forward(self, step, mb, vs):
        t0 = time.monotonic()
        first = vs == 0
        if first:
            x, _ = make_batch(
                self.spec.data_seed,
                step,
                mb,
                self.spec.mb_size,
                self.spec.model.seq_len,
                self.spec.model.vocab_size,
                dataset=self.spec.dataset_path,
                data_shards=self.spec.data_shards,
            )
            inp = x.to(self.device)
            if self._tied_role == "embed":
                self._tied_install(step)
        else:
            inp = self.fwd_ready.pop((step, mb, vs)).to(self.device).requires_grad_(True)

        if self.spec.recomputes(self.rank):
            # first stage recomputes too: its boundary is the token batch
            # (no input grad needed), holding the graph would keep every
            # in-flight microbatch's activations alive
            with torch.no_grad(), self._compute_ctx():
                out = self._layers(step, mb, vs, inp)
            self.ctx[(step, mb, vs)] = ("boundary", inp)
        else:
            with self._compute_ctx():
                out = self._layers(step, mb, vs, inp)
            self.ctx[(step, mb, vs)] = ("graph", inp, out)
        if self.moe:
            self._record_moe(step, mb, vs)

        self._sim_slow_wait(step, t0)
        self._ev("fwd", step, mb, vs, t0)
        if vs != self.final_vs:
            self._route_fwd(step, mb, vs + 1, out.detach())

    def _rebuild(self, step, mb, vs):
        kind, *rest = self.ctx.pop((step, mb, vs))
        if kind == "graph":
            inp, out = rest
            if self.moe:
                self._moe_f.pop((step, mb))
            return inp, out
        (inp,) = rest
        # The same autocast region as the forward: recompute exists to reproduce
        # that forward exactly, and a different precision would hand backward
        # the gradients of a function the forward never evaluated, silently.
        with self._compute_ctx():
            out = self._layers(step, mb, vs, inp)
        if self.moe:
            self._replay_moe(step, mb, vs)
        return inp, out

    def _layers(self, step, mb, vs, out):
        """The chunk's layers on `out`; under shard "pieces" each layer first
        waits for the step-`step` copy of the units it reads (RUNTIME-I13).
        arch "hf" units also take the microbatch's ids, recomputed on this
        stage's device (BA-407, BA-408): no index travels."""
        if self.hf:
            from wantrain.model.gpt import call_layer

            ids = self._batch(step, mb)[0].to(self.device)
            for li, layer in enumerate(self.chunks[vs]):
                if self.pieces_mode:
                    self._await_layer(step, li)
                out = call_layer(layer, out, ids)
            return out
        ready = self._opt_ready
        for li, layer in enumerate(self.chunks[vs]):
            if self.pieces_mode:
                self._await_layer(step, li)
            if ready is not None:
                self._await_opt_layer(ready, li)
            out = layer(out)
        return out

    def _batch(self, step, mb):
        s = self.spec
        return make_batch(
            s.data_seed,
            step,
            mb,
            s.mb_size,
            s.model.seq_len,
            s.model.vocab_size,
            dataset=s.dataset_path,
            data_shards=s.data_shards,
        )

    def _backward_final(self, step, mb, vs):
        t0 = time.monotonic()
        inp, out = self._rebuild(step, mb, vs)
        _, tgt = make_batch(
            self.spec.data_seed,
            step,
            mb,
            self.spec.mb_size,
            self.spec.model.seq_len,
            self.spec.model.vocab_size,
            dataset=self.spec.dataset_path,
            data_shards=self.spec.data_shards,
        )
        # Parity with the built-in heads, which all return logits.float():
        # cross-entropy over bf16 logits loses precision in the one number
        # the run is judged by. HF logits: loss_fn(logits.float(), ...) (BA-410).
        if self._user_autocast or self.hf:
            out = out.float()
        loss = loss_fn(out, tgt.to(self.device))
        self.losses[(step, mb)] = loss.item()
        self._stream_loss(step, mb, loss.item())
        self._run_backward(step, mb, vs, loss, None, inp, t0)

    def _backward_mid(self, step, mb, vs):
        t0 = time.monotonic()
        g = self.bwd_ready.pop((step, mb, vs)).to(self.device)
        inp, out = self._rebuild(step, mb, vs)
        self._run_backward(step, mb, vs, out, g, inp, t0)

    def _run_backward(self, step, mb, vs, out, gout, inp, t0):
        """B now, W later. B walks the activation-gradient chain: the input
        gradient goes upstream immediately, the non-linear parameters (norms,
        embeddings) get theirs, and every split linear leaves its dW pending in
        the store; the graph is freed here. W, run where the schedule puts it,
        executes the pending dW GEMMs and folds the microbatch's gradients."""
        params = self._chunk_params(vs)
        owned = self._wgrad_owned[vs]
        others = [p for p in params if id(p) not in owned]
        need_input_grad = vs != 0
        wc = [self._tied] if self._tied_role == "embed" else []
        inputs = ([inp] if need_input_grad else []) + others + wc
        with wgrad.capture() as store:
            if inputs:
                gs = list(torch.autograd.grad(out, inputs, grad_outputs=gout, allow_unused=True))
            else:
                torch.autograd.backward(out, grad_tensors=gout)
                gs = []
        if need_input_grad:
            self._route_bwd(step, mb, vs - 1, gs.pop(0))
        if wc:
            self._send_tied_embed(step, mb, gs.pop())
        other_grads = {id(p): g for p, g in zip(others, gs)}
        if self._streams_fold():
            # RUNTIME-I21: B's gradients fold now instead of waiting for W (a
            # dense embedding gradient per pending microbatch otherwise).
            a = self._step_acc(step, vs)
            for i, p in enumerate(params):
                g = other_grads.pop(id(p), None)
                if g is not None:
                    a.fold_part(mb, i, len(params), g)
        self._sim_slow_wait(step, t0)
        self._ev("bwd_b", step, mb, vs, t0)

        def w_pass(step=step, mb=mb, vs=vs, params=params, store=store, other_grads=other_grads):
            tw = time.monotonic()
            if self.pieces_mode:
                self._w_pieces(step, mb, params, store, other_grads)
            elif self.native and self.wide and self.w > 1:
                self._w_native(step, mb, vs, params, store, other_grads)
            else:
                # Width 1 keeps flush under every wire (BA-46): nothing to stream.
                dws = wgrad.flush(store)
                gw = []
                for p in params:
                    g = other_grads[id(p)] if id(p) in other_grads else dws.get(id(p))
                    if g is None:
                        raise RuntimeError("a trainable parameter received no gradient in B/W")
                    gw.append(g)
                self._store_grads(step, mb, vs, gw)
            self._sim_slow_wait(step, tw)
            self._ev("bwd_w", step, mb, vs, tw)

        if self._streams_fold() and not (self.native and self.wide and self.w > 1):
            # RUNTIME-I27: the same W, runnable in sub-blocks when it fills a wait (_fill_w)
            self.deferred_w.append((step, mb, vs, self._w_task(step, mb, vs, params, store)))
        else:
            self.deferred_w.append((step, mb, vs, w_pass))

    def _w_task(self, step, mb, vs, params, store):
        """One microbatch's streamed W (RUNTIME-I21) as a _WTask (RUNTIME-I27)."""
        return _WTask(self, step, mb, vs, params, store)

    def _streams_fold(self) -> bool:
        """RUNTIME-I21: a width-1 slot folds each gradient as it is produced,
        B's at B and W's at W (OrderedAccumulator.fold_part), so neither a
        pending microbatch's B gradients nor the stage's whole W set is
        resident beside the running sum. A ZeRO-1 or pieces stage and the
        Head of a split tied table (its row fold) keep their own paths."""
        return self.w == 1 and not self.pieces_mode and self._tied_role != "head"

    def _step_acc(self, step, vs) -> OrderedAccumulator:
        a = self.acc.get((step, vs))
        if a is None:
            a = self.acc[(step, vs)] = OrderedAccumulator(range(self.M))
        return a

    def _store_grads(self, step, mb, vs, gw):
        self.done_mbs.setdefault(step, set()).add((mb, vs))
        if self.wide and self.w > 1:
            if self.spec.stream_grads:
                # overlap the ZeRO-1 exchange with the backward tail: ship
                # this microbatch's shard slices to their owners immediately;
                # fold the own-shard slice without retaining it
                gw = list(gw)
                for k in range(self.w):
                    if k == self.peer:
                        if self._biases:
                            self._count_add(step, mb, self._counts_for(step, mb, k), self.peer)
                        self._shard_add(step, mb, gw[self.peer :: self.w])
                    else:
                        msg = ("mb", mb, [t.detach().cpu() for t in gw[k :: self.w]])
                        if self._biases:
                            # RUNTIME-I18 (BA-454): the counts of k's biases ride along
                            msg = (*msg, self._counts_for(step, mb, k))
                        self.send_qs[("intra", k)].put((fr.GRADS, step, mb, 0, msg, OBJ))
            else:
                # batch exchange retains this peer's own contributions until
                # the step barrier — bounded by M/w, the scalable mode is
                # stream_grads
                self.grads.setdefault(step, {})[(mb, vs)] = gw
        else:
            if self._tied_role == "head":
                # RUNTIME-I19: the tied table folds by rows (TiedRowFold), the
                # rest by microbatch index.
                gw = list(gw)
                h = gw.pop(self._tied_idx)
                self._tied_fold(step).add_head(mb, h, self._tied_ids(step, mb))
            a = self.acc.get((step, vs))
            if a is None:
                a = self.acc[(step, vs)] = OrderedAccumulator(range(self.M))
            a.add(mb, gw)

    def _shard_add(self, step, mb, tensors):
        """Fold one microbatch's own-shard contribution (local or remote)
        into the owner-side accumulator, in fixed index order."""
        a = self.shard_acc.get(step)
        if a is None:
            a = self.shard_acc[step] = OrderedAccumulator(range(self.M))
        a.add(mb, [t.to(self.device) for t in tensors])

    # ---------------- optimizer sync ----------------

    def _optimizer_step(self, step):
        if self.wide and self.w > 1:
            self._optimizer_step_wide(step)
        else:
            self._optimizer_step_local(step)

    def _optimizer_step_local(self, step):
        """Apply the incrementally-folded fixed-order sums, then AdamW."""
        t0 = time.monotonic()
        flat = []
        for vs in self.my_vs:
            flat += self.acc.pop((step, vs)).result()
        if self._tied_role == "head":
            fold = self._tied_fold(step)
            while not fold.complete():
                self._pump(block=True)
            flat.insert(self._tied_idx, self._tfold.pop(step).result())
        for p, g in zip(self.params, flat):
            p.grad = g
        if self.opt is not None and self._overlaps_opt():
            self._start_opt_overlap(step)
        elif self.opt is not None:  # stage 0 holding only a split tied table
            self._promote_moments()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        if self.moe:
            self._moe_bias_step(step, self.chunks[self.my_vs[0]])
        self.done_mbs.pop(step, None)
        self._ev("step_opt", step, -1, -1, t0)

    # ---------------- RUNTIME-I23: the optimizer overlapped with the next step ----------------

    def _overlaps_opt(self) -> bool:
        """Host-resident moments make the step's AdamW a PCIe round trip of every parameter (about 0.8 s per 1B on a
        PCIe 4.0 x16 H100 host); with nothing else coupled to it, it runs beside the next step instead of before it.
        Width-1 builtin stages only: ZeRO-1 exchanges, pieces, a tied table's relay and MoE bias updates are tied
        to the step barrier."""
        return (
            self.spec.optimizer == "adamw-host"
            and self.w == 1
            and not self.pieces_mode
            and not self.moe
            and not self.hf
            and self._tied_role is None
            and len(self.my_vs) == 1
        )

    def _start_opt_overlap(self, step: int = -1) -> None:
        """Run this step's AdamW on a side thread (a side CUDA stream on GPU), parameter by parameter in chunk order;
        chunk layer li is released (`_opt_ready[li]`) once every parameter it owns holds the update, and the next
        step's forward waits per layer (_layers). Same ops, same order, so the same bits as stepping first."""
        self._join_opt()
        self._promote_moments()
        chunk = self.chunks[self.my_vs[0]]
        owner: dict = {}
        for li, layer in enumerate(chunk):
            for p in layer.parameters():
                if p.requires_grad:
                    owner.setdefault(id(p), li)
        layer_of = [owner[id(p)] for p in self.params]
        left = [0] * len(chunk)
        for li in layer_of:
            left[li] += 1
        ready = [threading.Event() for _ in chunk]
        cuda = self.device.type == "cuda"
        # On CUDA a layer is released by an event on the side stream, which the forward's stream waits on: neither
        # the optimizer thread nor the forward's blocks on the other's GPU work.
        events = [None] * len(chunk)
        for li, n in enumerate(left):
            if n == 0:
                ready[li].set()
        # One side stream for the run: the caching allocator keeps freed blocks per stream, so a new stream each step
        # left every step's optimizer temporaries cached where no later step could reuse them (reserved memory grew
        # ~0.35 GiB a step on a 24 GB card while allocated memory stayed flat, 2026-10-13).
        if cuda and self._opt_stream is None:
            self._opt_stream = torch.cuda.Stream(device=self.device)
        side = self._opt_stream if cuda else None
        if side is not None:
            side.wait_stream(torch.cuda.current_stream(self.device))
            for p in self.params:
                if p.grad is not None:
                    p.grad.record_stream(side)  # freed by zero_grad on the side thread
        errors: list = []

        def release(i):
            li = layer_of[i]
            left[li] -= 1
            if left[li] == 0:
                if side is not None and os.environ.get("WANTRAIN_OPT_RELEASE") == "sync":
                    side.synchronize()  # diagnosis: RUNTIME-I23's host-side release
                elif side is not None:
                    ev = torch.cuda.Event()
                    ev.record(side)  # the layer's parameters are written before anyone reads them
                    events[li] = ev
                ready[li].set()

        t0 = time.monotonic()

        def work():
            try:
                ctx = torch.cuda.stream(side) if side is not None else contextlib.nullcontext()
                with ctx:
                    self.opt.step(on_param=release)
                    self.opt.zero_grad(set_to_none=True)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            finally:
                for ev in ready:
                    ev.set()
                self._ev("opt_bg", step, -1, -1, t0)

        self._opt_error = errors
        self._opt_ready = ready
        self._opt_events = events
        self._opt_thread = threading.Thread(target=work, name="opt-overlap", daemon=True)
        self._opt_thread.start()
        if not self._opt_logged:
            self._opt_logged = True
            print(f"WT_OPT overlap layers={len(chunk)} params={len(self.params)}", flush=True)

    def _await_opt_layer(self, ready, li) -> None:
        ready[li].wait()
        if self._opt_error:
            raise RuntimeError(f"overlapped optimizer step failed: {self._opt_error[0]!r}")
        evs = self._opt_events
        if evs is not None and evs[li] is not None:
            torch.cuda.current_stream(self.device).wait_event(evs[li])

    def _promote_moments(self) -> None:
        """RUNTIME-I24: before this process's third optimizer step, move host moments onto the device into the
        room the first two steps left: 85% of the memory free before the build (the fit check's budget) less the
        reserved peak and a tenth of the card (at least 1 GiB). A resident parameter's update skips the PCIe round trip; the values and
        the ops are the same wherever the moments live, so the bits are too. WANTRAIN_OPT_RESIDENT=off keeps
        every moment on the host."""
        self._opt_steps += 1
        if (
            self._opt_steps != 3
            or self.device.type != "cuda"
            or not hasattr(self.opt, "promote")
            or os.environ.get("WANTRAIN_OPT_RESIDENT", "auto") == "off"
        ):
            return
        peak = int(torch.cuda.max_memory_reserved(self.device))
        # A margin of a tenth of the card (at least 1 GiB): the peak is the last step's (_print_mem resets it), and
        # a later step can run above it.
        room = int(0.85 * self._free_at_init) - peak - max(1 << 30, self._free_at_init // 10)
        if self.opt is None:
            return
        moved = self.opt.promote(room) if room > 0 else 0
        n = len(getattr(self.opt, "resident", ()))
        print(
            f"WT_OPT resident bytes={moved} params={n}/{len(self.params)} room={room} peak={peak}",
            flush=True,
        )

    def _join_opt(self) -> None:
        """Wait for an overlapped optimizer step (before a checkpoint reads the state, the next step, the end)."""
        t = self._opt_thread
        if t is not None:
            t.join()
            self._opt_thread = None
            self._opt_ready = None
            self._opt_events = None
            if self._opt_error:
                raise RuntimeError(f"overlapped optimizer step failed: {self._opt_error[0]!r}")

    def _optimizer_step_wide(self, step):
        """ZeRO-1 exchange. Peer k owns params[k::w]. Every peer ships its
        per-microbatch shard gradients to each owner; owners sum all M
        microbatches in fixed index order, update their shard, and broadcast
        the new shard parameters. With stream_grads the shipping happened
        during the backward tail and this only waits for completeness."""
        t0 = time.monotonic()
        vs = self.my_vs[0]
        exec_mbs = self.my_mbs
        if self.native:
            # one ordered fold per owned stride tensor, widened contributions
            owned = range(self.peer, len(self.params), self.w)

            def folded():
                return all(
                    (step, i) in self._native_acc and self._native_acc[(step, i)].complete()
                    for i in owned
                )

            while not folded():
                self._pump(block=True)
            acc = [self._native_acc.pop((step, i)).result()[0] for i in owned]
        elif self.spec.stream_grads:
            # contributions were folded as they arrived; wait for completeness
            while not (step in self.shard_acc and self.shard_acc[step].complete()):
                self._pump(block=True)
            acc = self.shard_acc.pop(step).result()
        else:
            local = {mb: self.grads[step][(mb, vs)] for mb in exec_mbs}
            for k in range(self.w):
                if k == self.peer:
                    if self._biases:
                        for mb in exec_mbs:
                            self._count_add(step, mb, self._counts_for(step, mb, k), self.peer)
                    continue
                msg = {mb: [t.detach().cpu() for t in local[mb][k :: self.w]] for mb in exec_mbs}
                if self._biases:
                    # RUNTIME-I18 (BA-454): the counts of k's biases, per microbatch
                    msg["counts"] = {mb: self._counts_for(step, mb, k) for mb in exec_mbs}
                self.send_qs[("intra", k)].put((fr.GRADS, step, 0, 0, msg, OBJ))
            # fold own + arriving contributions in fixed index order
            folder = OrderedAccumulator(range(self.M))
            for mb in exec_mbs:
                folder.add(mb, local[mb][self.peer :: self.w])
            while not folder.complete():
                self._pump(block=True)
                for msg in self._sync_grads.values():
                    for mb, tensors in msg.items():
                        folder.add(mb, [t.to(self.device) for t in tensors])
                self._sync_grads = {}
            acc = folder.result()
        for p, g in zip(self.shard_params, acc):
            p.grad = g
        if self.opt is not None:  # an owner-less peer only exchanges
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        if self._my_bias:
            # every owned bias's counts of every microbatch, then its update
            fold = self._count_fold(step)
            while self.native and not fold.complete():
                self._pump(block=True)
            self._moe_bias_step(step, self.chunks[vs])
        else:
            self._moe_counts.pop(step, None)
        # broadcast updated shard params (and owned biases), collect the others'
        fresh = [p.detach().cpu().clone() for p in self._own[self.peer :: self.w]]
        for k in range(self.w):
            if k == self.peer:
                continue
            self.send_qs[("intra", k)].put((fr.PARAMS, step, 0, 0, (self.peer, fresh), OBJ))
        while len(self._sync_params) < self.w - 1:
            self._pump(block=True)
        with torch.no_grad():
            for owner, tensors in self._sync_params.items():
                for p, t in zip(self._own[owner :: self.w], tensors):
                    p.copy_(t.to(self.device))
        self._sync_params = {}
        self.grads.pop(step, None)
        self.done_mbs.pop(step, None)
        self._ev("step_opt", step, -1, -1, t0)

    # ---------------- main loop ----------------

    def run(self):
        # print() resolves sys.stdout per call, so the loss tail's thread
        # writes through this wrapper too.
        if not isinstance(sys.stdout, LineAtomicStdout):
            sys.stdout = LineAtomicStdout(sys.stdout)
        # A data-reading slot holds, verified, the shards its microbatches draw before it connects:
        # a missing or different corpus refuses here, not at the first microbatch (model/provision.py).
        provision.provision_slot(self.spec, self.rank, self.peer)
        self._connect()
        spec = self.spec
        # Resume after connect: loading a shard takes seconds and the listeners
        # must already be up, or a fast peer dials a rank still deserializing.
        start_step = self._resume() if spec.resume else 0
        self._drop_abandoned_waves(start_step)
        if self.pieces_mode:
            self._init_copies(start_step)
        if self._relay:
            self._tied_plan(start_step)
            if self._tied_role == "head" and spec.resume and start_step < spec.steps:
                # a resume sends R_k from the loaded table before its first step
                self._send_tied_rows(start_step)
        t_first = None
        try:
            for step in range(start_step, spec.steps):
                t_step = time.perf_counter()
                if t_first is None:
                    t_first = self._t_first = time.monotonic_ns()
                self._exec_step = step
                if self.pieces_mode:
                    self._step_t0[step] = t_step
                self._run_step(step)
                if self.deferred_w:
                    raise RuntimeError(f"step {step} ended with {len(self.deferred_w)} W pending")
                if self.pieces_mode:
                    # No optimizer barrier: each unit updates in the first pump
                    # after its fold (RUNTIME-I13); the full barrier only at
                    # checkpoint steps and the last step (BA-9, BA-10).
                    self._ran_step = step
                    if self._is_barrier(step):
                        self._pieces_barrier(step)
                        self._emit_ready_reports()
                        ck = bool(self.spec.ckpt_every and (step + 1) % self.spec.ckpt_every == 0)
                        self._report_step(step, checkpoint=ck)
                    else:
                        self._pump(block=False, top=True)
                    continue
                self._optimizer_step(step)
                if self._tied_role == "head" and tied_sends_after(step, spec.steps - 1):
                    self._send_tied_rows(step + 1)
                if self._seen:
                    # duplicate keys of finished steps can no longer pass the step check
                    self._seen = {k for k in self._seen if k[1] > step}
                self._emit_step_telemetry(step, (time.perf_counter() - t_step) * 1e3)
                if self.native and self.wide and self.w > 1:
                    self.holds.end_step(step)
                    print(self.holds.line(step), flush=True)
                self._print_mem(step)
                if self.spec.ckpt_every and (step + 1) % self.spec.ckpt_every == 0:
                    self._join_opt()  # the checkpoint reads the step's updated state
                    self._checkpoint(step)
                # Every slot reports its own progress (only the last stage has a
                # loss to report), so a supervisor can see which slot stopped.
                print(f"WT_STEP {step}", flush=True)
                # Monotonic stamps only: no tensor op, no device sync (WT_AEAD).
                self._steps_run += 1
                self._steps_wall_ns = time.monotonic_ns() - t_first
            # The final checkpoint is unconditional: ckpt_every only gates the
            # periodic saves, and a completed run owes its weights regardless.
            self._join_opt()
            self._checkpoint(self.spec.steps - 1)
            self._await_wave()
        except BaseException:
            t = self._opt_thread
            if t is not None:
                t.join()
            self._settle_wave()
            # A failed run closes its peer sockets as _finish does: the
            # neighbours see the end now, and nothing is left to the GC.
            for sk in {id(x): x for x in self.conns.values()}.values():
                _hard_close(sk)
            raise
        self._finish()

    def _print_mem(self, step):
        if self.device.type == "cuda":
            # One line per step, so a memory surprise shows up as a growth
            # curve in the log rather than a single OOM message.
            print(
                f"WT_MEM step={step} "
                f"alloc={torch.cuda.memory_allocated(self.device) / 2**30:.2f} "
                f"peak={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f} "
                f"reserved={torch.cuda.memory_reserved(self.device) / 2**30:.2f}",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def _run_w(self, step, mb, vs):
        """Run the deferred dL/dW of one microbatch, wherever it sits in the
        battery: a schedule names the microbatch, not the oldest pending."""
        for i, (s_, mb_, vs_, cl) in enumerate(self.deferred_w):
            if (s_, mb_, vs_) == (step, mb, vs):
                del self.deferred_w[i]
                cl()
                return
        raise RuntimeError(f"schedule asks for W of step {step} mb {mb} before its B")

    # RUNTIME-I27: a W filling a wait checks the inbox this often (WANTRAIN_TEST_W_SLICE_S: timing only, 0 = every
    # gradient its own sub-block)
    W_SLICE_S = float(os.environ.get("WANTRAIN_TEST_W_SLICE_S", "0.005"))

    def _fill_w(self, step, vs, order, i) -> bool:
        pending = {(s_, mb_, vs_): cl for s_, mb_, vs_, cl in self.deferred_w}
        for j in range(i + 1, len(order)):
            op, mb = order[j]
            if op == "W" and (step, mb, vs) in pending:
                task = pending[(step, mb, vs)]
                if not hasattr(task, "run"):
                    del order[j]
                    self._run_w(step, mb, vs)
                    return True
                # RUNTIME-I27: in sub-blocks, the inbox read between them, so the op the order waits on starts as soon
                # as its input is here instead of after the whole W; what is left stays in the order.
                while not task.run(time.monotonic() + self.W_SLICE_S):
                    self._w_slices += 1
                    self._pump(block=False, top=True)
                    if self._runnable(step, vs, order[i]):
                        return True
                del order[j]
                self._drop_w(step, mb, vs)
                return True
        return False

    def _runnable(self, step, vs, item) -> bool:
        """Whether op `item` of the order can start now (the step loop's own tests)."""
        op, mb = item
        if op == "F":
            return (vs == 0 and self._tied_ok(step, mb)) or (step, mb, vs) in self.fwd_ready
        if op == "B":
            return vs == self.final_vs or (step, mb, vs) in self.bwd_ready
        return True

    def _drop_w(self, step, mb, vs):
        for k, (s_, mb_, vs_, _cl) in enumerate(self.deferred_w):
            if (s_, mb_, vs_) == (step, mb, vs):
                del self.deferred_w[k]
                return

    def _run_step(self, step):
        """Execute this slot's zero-bubble order (scheduler/zero_bubble.py):
        each op runs as soon as its input is here, in the order the scheduler
        solved for the measured costs and links. Gradients fold in microbatch
        order, so any valid order gives the same bits."""
        vs = self.my_vs[0]
        order = [(op, int(mb)) for op, mb in self.spec.schedule_order[self.rank][self.peer]]
        pad = self._slot_wait if self._slot_pad is not None else lambda step, op, mb: None
        i = 0
        while i < len(order):
            op, mb = order[i]
            self._pump(block=False, top=True)
            if op == "F":
                if (vs == 0 and self._tied_ok(step, mb)) or (step, mb, vs) in self.fwd_ready:
                    pad(step, op, mb)
                    self._forward(step, mb, vs)
                    i += 1
                    continue
            elif op == "B":
                if vs == self.final_vs:
                    # the loss lives here: B needs only this stage's own F
                    pad(step, op, mb)
                    self._backward_final(step, mb, vs)
                    i += 1
                    continue
                if (step, mb, vs) in self.bwd_ready:
                    pad(step, op, mb)
                    self._backward_mid(step, mb, vs)
                    i += 1
                    continue
            else:
                pad(step, op, mb)
                self._run_w(step, mb, vs)
                i += 1
                continue
            # Blocked on the network. The plan was solved for estimated costs
            # and a nominal link; when an input is late, a W the plan places
            # further on is already runnable and needs nothing from anyone, so
            # pulling it forward fills the wait without breaking a dependency.
            if self.spec.schedule_fill and self._fill_w(step, vs, order, i):
                continue
            t0 = time.monotonic()
            self._pump(block=True, top=True)
            self._ev("stall", step, -1, -1, t0)

    def _slot_wait(self, step, op, mb):
        """WANTRAIN_TEST_SLOT_PAD (timing only, slot_pad_seam): hold `op` of
        microbatch `mb` until its time in the mb's slot, pumping the inbox
        meanwhile so folds, updates and gathers go on. The slot origin is when
        the step's slot-0 F began plus the gather waits inside it."""
        tau, cb, cw = self._slot_pad
        i = self._slot_of[mb]
        if op == "F" and i == 0:
            self._pad_t0 = (step, time.monotonic(), self._gather_wait)
            return
        if self._pad_t0 is None or self._pad_t0[0] != step:
            raise RuntimeError(
                f"WANTRAIN_TEST_SLOT_PAD: step {step} runs {op} of microbatch {mb} before the F "
                f"of its first owned microbatch"
            )
        _, t0, w0 = self._pad_t0
        origin = t0 + self._gather_wait - w0
        due = origin + {"F": i * tau, "B": (i + 1) * tau - cw - cb, "W": (i + 1) * tau - cw}[op]
        while (left := due - time.monotonic()) > 0:
            self._pump(block=True, top=True, timeout=left)

    def _pump(self, block, top=False, timeout=1.0):
        """Drain the inbox (waiting up to `timeout` s, default 1, for a first
        item when `block`). Under shard "pieces" every pump then runs the unit updates whose fold
        completed (RUNTIME-I13; WANTRAIN_TEST_UPDATES=between_ops: only the
        non-blocking pumps between ops, a timing-only liveness control), and a
        top-level pump (the step loop's, never one nested inside an op)
        reports the steps whose updates are all applied."""
        updates = self.pieces_mode and not (self._updates_between_ops and block)
        # A pump may run nested inside an op's autocast or grad region; the
        # exchange (folds, updates, casts, installs) never runs inside one.
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            # A fold completed in an op (an own last contribution) updates now,
            # never after the inbox's 1 s wait (a width-1 stage's barrier).
            self._pump_inbox(block and not (updates and self._ready), timeout)
            if updates:
                self._run_updates()
        if self.pieces_mode and top:
            self._emit_ready_reports()

    def _pump_inbox(self, block, timeout=1.0):
        try:
            while True:
                item = self.inbox.get(block=block, timeout=timeout if block else None)
                block = False
                kind = item[0]
                if kind == "fatal":
                    # Raised here, on the main thread, so run() unwinds, the
                    # launcher prints WT_DONE with a nonzero code, and the
                    # orchestrator can kill the fleet instead of billing it.
                    raise RuntimeError(item[1])
                _, f, body, src = item
                # "net_dec"/"net_obj": the reader thread already decoded the
                # body, so the main loop pays a dict insert, not an inflate.
                vs = self.my_vs[0]
                if kind == "net_bad":
                    raise RuntimeError(f"refused frame type {f.type} from {src}: envelope: {body}")
                if f.type == fr.FWD:
                    if self.wide:
                        # chunk carries the sending peer for B routing
                        self.fwd_origin[(f.step, f.mb)] = f.chunk
                    self.fwd_ready[(f.step, f.mb, vs)] = (
                        body if body is not None else self._decode_body(f)
                    )
                elif f.type == fr.BWD:
                    self.bwd_ready[(f.step, f.mb, vs)] = (
                        body if body is not None else self._decode_body(f)
                    )
                elif f.type in (fr.GRADS, fr.PARAMS):
                    try:
                        self._on_exchange(f, body, src)
                    except xc.FrameError as e:
                        name = "GRADS" if f.type == fr.GRADS else "PARAMS"
                        raise RuntimeError(f"refused {name} frame from {src}: {e}") from e
                elif f.type == fr.LOSS:
                    self._peer_losses.append(f.obj())
                elif f.type == getattr(fr, "TIED", None):
                    self._on_tied(f, src)
                else:
                    # BA-416: a frame type this loop does not handle is never
                    # dropped silently.
                    raise RuntimeError(f"refused frame type {f.type} from {src}: not handled here")
        except queue.Empty:
            pass

    def _report_aead(self):
        """BA-44: the sealing cost of this slot, summed over every bound
        SecureSocket, next to the wall time of the steps it ran."""
        tot = dict(seal_ns=0, open_ns=0, bytes_out=0, bytes_in=0)
        for ss in self._secure:
            st = ss.stats()
            for k in tot:
                tot[k] += st[k]
        print(f"WT_WFILL rank={self.rank} peer={self.peer} slices={self._w_slices}", flush=True)
        print(
            f"WT_AEAD rank={self.rank} peer={self.peer} seal_ns={tot['seal_ns']} "
            f"open_ns={tot['open_ns']} bytes_out={tot['bytes_out']} bytes_in={tot['bytes_in']} "
            f"steps={self._steps_run} steps_wall_ns={self._steps_wall_ns}",
            flush=True,
        )

    def _finish(self):
        os.makedirs(self.spec.out_dir, exist_ok=True)
        tag = self.spec.tag
        base = os.path.join(self.spec.out_dir, f"{tag}_rank{self.rank}p{self.peer}")
        self._flush_telemetry()
        # device-map fingerprint: reproducibility is claimed under this map,
        # so replay comparisons must match it
        fp = {"torch": torch.__version__, "device": str(self.device)}
        if self.device.type == "cuda":
            fp["gpu"] = torch.cuda.get_device_name(self.device)
            fp["cuda"] = torch.version.cuda
        with open(base + "_env.json", "w") as f:
            json.dump(fp, f, indent=2)
        if self.hosts_final:
            if self.w > 1 and self.peer != 0:
                self.send_qs[("intra", 0)].put(
                    (fr.LOSS, 0, 0, 0, {f"{s}:{m}": v for (s, m), v in self.losses.items()}, OBJ)
                )
            else:
                # Still live: a peer that dies before reporting its losses is a
                # failure, and the connection loss must surface as one.
                need = self.w - 1
                while len(self._peer_losses) < need:
                    self._pump(block=True)
                for msg in self._peer_losses:
                    for k, v in msg.items():
                        s, m = k.split(":")
                        self.losses[(int(s), int(m))] = v
                out = {
                    "losses": [
                        {"step": s, "mb": m, "loss_hex": float_hex(v), "loss": v}
                        for (s, m), v in sorted(self.losses.items())
                    ]
                }
                with open(os.path.join(self.spec.out_dir, f"{tag}_losses.json"), "w") as f:
                    json.dump(out, f, indent=2)
        self._drain_tied_relay()
        # From here on a closed connection is the expected end of a run.
        self._closing = True
        # Drain, then close: the sender paces to the measured link rate, so the
        # last gradient may still be mid-flight when the loop ends. The sentinel
        # travels the whole sender pipeline (engine queue, encoder, fragment
        # senders); each sender exits only after sending what was queued before
        # it, then sends END and half-closes. Queues are deduped by identity:
        # keys may share one, and a second sentinel would stop one sender twice.
        for q in {id(q): q for q in self.send_qs.values()}.values():
            q.put(None)
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        for t in self._send_threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        # Then wait for every neighbour's END (or its EOF): until then it may
        # still be writing to us, and closing early would reset its connection.
        for t in self._read_threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        # Sends that completed during the drain belong in the event log too.
        self._flush_telemetry()
        stuck = [t for t in self._send_threads + self._read_threads if t.is_alive()]
        if stuck:
            # Bounded on purpose: a wedged pacer or a silent neighbour must not
            # hold a box hostage. Say so and close anyway.
            print(
                f"WT_WARN shutdown: {len(stuck)} wire thread(s) still busy "
                f"after {SHUTDOWN_GRACE_S:.0f}s; closing sockets anyway",
                flush=True,
            )
        self._report_aead()
        if self.copy is not None:
            # RUNTIME-I13 on CPU: every exchange copy ran on the copy engine's
            # thread, none on this (compute) thread (thread identity).
            self.copy.close()
            print(
                f"WT_COPY rank={self.rank} peer={self.peer} copies={self.copy.copies} "
                f"engine_threads={sorted(self.copy.ran_on)} compute_thread={threading.get_ident()}",
                flush=True,
            )
        for b in self._budgets.values():
            b.close()
        for sk in {id(x): x for x in self.conns.values()}.values():
            try:
                sk.close()
            except OSError:
                pass


# A state.pt serialized on the step, waiting for its background commit.
_STAGED_PREFIX = ".staged-step-"

# arch "hf": the run's buffer record (I-4) listed in every wave (RUNTIME-I19).
HF_BUFFERS = "hf_buffers.json"


def hf_buffers_bytes(model) -> bytes:
    """The buffer record pinned in the JobSpec, as canonical JSON bytes."""
    rec = model.hf_buffers
    if not isinstance(rec, dict):
        raise RuntimeError("arch 'hf' needs model.hf_buffers (the run's buffer record, I-4)")
    return json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()


def check_hf_buffers(wave_dir, pin) -> None:
    """A wave's hf_buffers.json against the pin (RUNTIME-I19, BA-417): every
    tensor's bytes hash to the pinned sha256; refused naming the tensor (or the
    file), before any state is applied."""
    path = os.path.join(wave_dir, HF_BUFFERS)
    try:
        with open(path, "rb") as f:
            rec = json.loads(f.read())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"resume refused: {path}: {type(e).__name__}: {e}") from None
    want = (pin or {}).get("tensors") if isinstance(pin, dict) else None
    have = rec.get("tensors") if isinstance(rec, dict) else None
    if not isinstance(want, dict) or not isinstance(have, dict):
        raise RuntimeError(f"resume refused: {path} or the pin holds no buffer tensors")
    for name in sorted(set(want) | set(have)):
        a, b = want.get(name), have.get(name)
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise RuntimeError(
                f"resume refused: BUFFER-PIN {name}: in only one of {path} and the pin"
            )
        try:
            raw = base64.b64decode(b.get("b64", ""), validate=True)
        except (ValueError, TypeError):
            raise RuntimeError(
                f"resume refused: BUFFER-PIN {name}: bytes not base64 in {path}"
            ) from None
        if hashlib.sha256(raw).hexdigest() != a.get("sha256") or b.get("sha256") != a.get("sha256"):
            raise RuntimeError(
                f"resume refused: BUFFER-PIN {name}: its bytes in {path} differ from the pin"
            )


def device_class_change(want, have) -> str | None:
    """The first slot "r<stage>p<peer>" whose device class differs between two
    device_classes maps (RUNTIME-I15's hf clause), or None."""
    if want == have:
        return None
    rows = max(len(want or []), len(have or []))
    for r in range(rows):
        a = (want or [])[r] if r < len(want or []) else []
        b = (have or [])[r] if r < len(have or []) else []
        for p in range(max(len(a or []), len(b or []))):
            x = a[p] if a and p < len(a) else None
            y = b[p] if b and p < len(b) else None
            if x != y:
                return f"r{r}p{p} ({y!r} -> {x!r})"
    return "the device_classes map"


def tied_sends_after(step: int, last_step: int) -> bool:
    """Whether the Head sends TIED(step + 1, k) after step's AdamW: only when
    step + 1 runs in this attempt, never after the attempt's last step (the
    run's last, or the step a checkpoint-and-stop, re-plan or restart cut the
    attempt's steps at), so END finds no TIED frame in flight."""
    return step < last_step


def check_tied_sparse(step, mb, e, outside) -> None:
    """TIED-SPARSE (RUNTIME-I19): stage 0's embedding gradient e_i must be +0
    (no bit set, -0 included) on every row outside T_i (`outside` mask). One
    read of e through an integer view of its bits (a row is all +0 exactly
    when its smallest and largest bit pattern are 0), O(rows) scratch: no
    copy of the rows outside T_i, which is nearly the whole table."""
    ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}[e.element_size()]
    if e.numel() == 0:
        return
    bits = e.detach().contiguous().view(ints).reshape(e.shape[0], -1)
    lo, hi = torch.aminmax(bits, dim=1)
    if bool((((lo != 0) | (hi != 0)) & outside).any()):
        raise RuntimeError(
            f"TIED-SPARSE (step {step}, mb {mb}): the embedding gradient has a nonzero or -0 "
            f"element outside the microbatch's ids"
        )


def moe_aux_hex(aux: dict) -> str:
    """One MoE layer's aux values of one (step, mb) as exact hex: canonical JSON
    of {name: float hex}, keys sorted. A value is a number or already a float
    hex string (arch.moe_forward records hex); both give the hex of the same
    double, the text validation.state.aux_hex gives the oracle's rows."""

    def h(v) -> str:
        return float.fromhex(v).hex() if isinstance(v, str) else float_hex(float(v))

    return json.dumps({k: h(v) for k, v in sorted(aux.items())}, separators=(",", ":"))


def moe_records(records, step, mb, layers) -> dict:
    """{layer: (aux hex, int64 counts)} from arch.take_moe_records' records of
    one forward: every MoE layer of the chunk exactly once (RUNTIME-I18)."""
    got: dict = {}
    for r in records:
        layer = int(r.layer)
        if layer in got:
            raise RuntimeError(f"MOE-RECORD (step {step}, mb {mb}, layer {layer}): recorded twice")
        got[layer] = (moe_aux_hex(r.aux), r.counts.detach().to("cpu", torch.int64))
    missing = sorted(set(layers) - set(got))
    if missing:
        raise RuntimeError(f"MOE-RECORD (step {step}, mb {mb}, layer {missing[0]}): not recorded")
    return got


def moe_replay_check(step, mb, recorded: dict, replayed: dict) -> None:
    """MOE-REPLAY (RUNTIME-I18): B's rebuild of microbatch mb must record, for
    every MoE layer, exactly F's aux hex and counts."""
    for layer in sorted(set(recorded) | set(replayed)):
        a, b = recorded.get(layer), replayed.get(layer)
        if (
            a is None
            or b is None
            or a[0] != b[0]
            or a[1].shape != b[1].shape
            or not torch.equal(a[1], b[1])
        ):
            raise RuntimeError(
                f"MOE-REPLAY (step {step}, mb {mb}, layer {layer}): the rebuild's aux or counts "
                f"differ from F's ({a[0] if a else None} vs {b[0] if b else None})"
            )


class _TiedSeq:
    """The (step, mb) frames of TIED one direction carries in an attempt.
    push(step, mb, frag_idx, frag_cnt) takes a fragment (a whole frame is
    (0, 1)). A step's frames may complete in any order (the fragments of a
    multi-stream link interleave across its sockets), but every frame of a
    step completes before any frame of the next step can be sent (the relay's
    data dependence: TIED(t+1, .) follows the Head's AdamW of step t, which
    needs every TIED(t, .)). Fatal: a frame of another step than the first
    incomplete frame's (out of sequence), one not of this attempt, one given
    again (duplicated), and a fragment inconsistent with its frame's count."""

    def __init__(self, expected):
        self.expected = list(expected)
        self.index = {key: n for n, key in enumerate(self.expected)}
        self.pos = 0  # the first incomplete frame
        self.done_keys: set = set()  # completed frames at or after pos
        self.open: dict = {}  # (step, mb) -> (fragment count, fragments seen)

    def push(self, step, mb, frag_idx=0, frag_cnt=1):
        key = (step, mb)
        if self.pos >= len(self.expected):
            raise RuntimeError(f"TIED (step {step}, mb {mb}) beyond the attempt's last frame")
        n = self.index.get(key)
        if n is None:
            raise RuntimeError(f"TIED (step {step}, mb {mb}) is no frame of this attempt")
        if n < self.pos or key in self.done_keys:
            raise RuntimeError(f"TIED (step {step}, mb {mb}) duplicated")
        want = self.expected[self.pos]
        if step != want[0]:
            raise RuntimeError(f"TIED (step {step}, mb {mb}) out of sequence, expected {want}")
        cnt, got = self.open.setdefault(key, (frag_cnt, set()))
        if frag_cnt != cnt or not 0 <= frag_idx < frag_cnt or frag_idx in got:
            raise RuntimeError(
                f"TIED (step {step}, mb {mb}) fragment {frag_idx}/{frag_cnt} refused"
            )
        got.add(frag_idx)
        if len(got) == cnt:
            del self.open[key]
            self.done_keys.add(key)
            while self.pos < len(self.expected) and self.expected[self.pos] in self.done_keys:
                self.done_keys.discard(self.expected[self.pos])
                self.pos += 1

    def done(self) -> bool:
        return self.pos == len(self.expected)

    def left(self):
        return self.expected[self.pos] if self.pos < len(self.expected) else None


class _RawFrag:
    """A received TIED fragment forwarded as is (its header fields and bytes)."""

    __slots__ = ("frame",)

    def __init__(self, frame):
        self.frame = frame


class _TiedRefused(Exception):
    """A reader thread refused a TIED fragment; the fatal is in the inbox."""


def _held_step(tmp):
    raise AssertionError("commit_wave wrote into a step the slot already holds")


def _seam_number(name, raw):
    try:
        x = float(raw)
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not a number") from None
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{name}: {raw!r} is not a finite number >= 0")
    return x


def intra_latency_seam(value) -> float:
    """WANTRAIN_TEST_INTRA_LATENCY_MS: a one-way delay (ms) on every
    intra-stage connection, through the shim's LinkSpec latency (E4's
    RTT/2 + 2f'/PCIe'). Timing only: a frame is delivered later, never
    another byte or another fold order. Unset or empty -> 0."""
    if value is None or value == "":
        return 0.0
    return _seam_number("WANTRAIN_TEST_INTRA_LATENCY_MS", value)


def slot_pad_seam(value, rank):
    """WANTRAIN_TEST_SLOT_PAD="rank:tau_ms:b_ms:w_ms": stage `rank` runs its
    i-th owned microbatch in the slot [o + i tau, o + (i + 1) tau) of its step
    (E4's scaled.stage0_padding): F not before the slot's start, B not before
    (i + 1) tau - w - b, W not before (i + 1) tau - w; o is when the step's
    first F began, plus its gather waits. Timing only: an op is held (the inbox
    pumped meanwhile), never given another input, output or order. -> (tau, b,
    w) in seconds for that stage, None for the others or when unset or empty."""
    if value is None or value == "":
        return None
    name = "WANTRAIN_TEST_SLOT_PAD"
    parts = value.split(":")
    if len(parts) != 4 or not parts[0].isdigit():
        raise ValueError(f"{name}={value!r}: expected rank:tau_ms:b_ms:w_ms")
    tau, b, w = (_seam_number(name, x) / 1e3 for x in parts[1:])
    if not tau > b + w:
        raise ValueError(f"{name}={value!r}: tau must exceed b + w")
    return (tau, b, w) if int(parts[0]) == rank else None


# Events a summary counts only for its own step (telemetry.summarize_step).
_STEP_EVENTS = frozenset(("fwd", "bwd_b", "bwd_w", "step_opt", "stall"))


def _hop(key) -> str:
    """Telemetry label of a connection key: ('next', 1) -> 'next1'."""
    return f"{key[0]}{key[1]}"


class _Refused(Exception):
    """An accepted data-plane socket refused (RUNTIME-I16): why, the slot it
    claimed and the id it proved ("-" where not known yet)."""

    def __init__(self, reason, st="-", pe="-", peer="-"):
        super().__init__(reason)
        self.reason, self.st, self.pe, self.peer = reason, st, pe, peer


# Refusals an I/O failure causes; once every slot is bound they are the
# acceptor's own closing of a socket still pending, reported as "unbound".
_IO_REASONS = ("auth", "silent", "hello")


def _hard_close(sk):
    with contextlib.suppress(OSError):
        sk.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sk.close()


class _Result:
    """Run fn on a thread; get() returns its value or re-raises its error."""

    def __init__(self, fn):
        self._val = None
        self._err = None

        def _run():
            try:
                self._val = fn()
            except BaseException as e:  # noqa: BLE001 -- re-raised in get()
                self._err = e

        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()

    def get(self):
        self._t.join()
        if self._err is not None:
            raise self._err
        return self._val

    def raise_if_failed(self):
        """Re-raise fn's error if it has already failed; never blocks."""
        if not self._t.is_alive() and self._err is not None:
            raise self._err


def _name_slot(msg, rt, step):
    """A refusal naming "slot pK" also names that slot's wave directory."""
    m = re.search(r"slot p(\d+)", msg)
    if m is None:
        return msg
    d = wave_dir(rt.spec.out_dir, rt.spec.tag, rt.rank, int(m.group(1)), step)
    return msg if d in msg else f"{msg} ({d})"


class _Exchange:
    """A queued GRADS or PARAMS chunk: its envelope, the copy engine's future
    of its host tensors, its tensor bytes and the hold counter it is charged to
    until sent."""

    __slots__ = ("env", "fut", "nbytes", "hold", "key")

    def __init__(self, env, fut, nbytes, hold, key):
        self.env, self.fut, self.nbytes, self.hold, self.key = env, fut, nbytes, hold, key


def _nbytes(tensors) -> int:
    return sum(int(t.numel()) * t.element_size() for t in tensors)


class _UnitFold:
    """One unit's piece fold for one step (RUNTIME-I11): the left fold over
    microbatches in index order, starting from an fp32 copy of microbatch 0's
    widened piece (pieces.fold_start), each later one widened and added
    (pieces.fold_add). An out-of-order contribution waits: a remote one in host
    memory (moved to the device by the copy engine only when it is folded), an
    own one on the device; both are charged to the hold counters until folded."""

    def __init__(self, n_mb, holds, h2d):
        self.n = n_mb
        self.pos = 0
        self.acc = None
        self.hold: dict = {}
        self.holds = holds
        self.h2d = h2d

    def add(self, mb, tensors, own):
        if mb in self.hold or mb < self.pos:
            raise RuntimeError(f"a second contribution of microbatch {mb} to one fold")
        nbytes = _nbytes(tensors)
        kind = "device_own" if own else "host_grads"
        if own and mb != self.pos:
            # The piece is a view of the whole unit gradient: held as a view it
            # would keep all w pieces resident, not the 1/w RUNTIME-I13 charges.
            tensors = [t.clone() for t in tensors]
            self.holds.add(kind, nbytes)
        elif own:
            nbytes = 0  # folded at once: never held
        self.hold[mb] = (tensors, own, nbytes, kind)
        while self.pos in self.hold:
            ts, is_own, nb, k = self.hold.pop(self.pos)
            if not is_own:
                ts = self.h2d(ts)
            if self.acc is None:
                self.acc = [pc.fold_start(t) for t in ts]
            else:
                for a, t in zip(self.acc, ts):
                    pc.fold_add(a, t)
            if nb:
                self.holds.sub(k, nb)
            self.pos += 1
        return self.pos == self.n
