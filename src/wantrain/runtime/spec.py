"""Job specification: the single JSON object that fully determines a run.

Workers are dumb executors of a spec; the coordinator (or a local launcher)
only ever ships one of these. Every numeric result is a pure function of it.

There is one execution model: zero-bubble pipeline parallelism. Every slot
runs a precomputed order of F (forward), B (input gradient, the critical
path) and W (weight gradient, deferred into the bubbles) for each
microbatch it owns. The order comes from scheduler/zero_bubble.py; the
coordinator solves it from costs profiled on each node and the measured
links, and `finalize()` solves one from the analytic cost model when a spec
arrives without one, so every spec is executable as is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from wantrain.model.gpt import GPTConfig
from wantrain.resources.node import mixed_fleet

OPS = ("F", "B", "W")
# The AdamW every stage and the reference run. A recipe pin: a shard written
# under other hyperparameters holds moments this run would not have produced.
ADAMW: dict[str, Any] = {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0}
_HEX = frozenset("0123456789abcdef")
WIRES = ("fp32", "native")
SHARDS = ("zero1", "pieces")
PACINGS = ("link", "shared")
# GPTConfig fields M4 added (HANDOUT-M4 I-1). A dense model's pins omit them
# (BA-465), so its pins stay byte-identical to the pre-M4 ones and a pre-M4
# dense wave still resumes; an hf model pins every hf_* field, an MoE model
# every moe_* field.
_HF_FIELDS = ("hf_config", "hf_buffers", "hf_census", "hf_code", "hf_impl")
# The rollback pieces (model/rollback.py, BA-467) whose target refuses a tied
# table split across stages: the row fold, the TIED relay, the planner's charge.
_TIED_PIECES = ("runtime-7a", "runtime-7b", "scheduler-6")


@dataclass
class LinkCfg:
    latency_ms: float = 0.0
    bw_mbps: float = 0.0
    jitter_ms: float = 0.0
    jitter_seed: int = 0


@dataclass
class JobSpec:
    # ------------------------------------------------------------- recipe
    model: GPTConfig = field(default_factory=GPTConfig)
    init_seed: int = 1234
    data_seed: int = 5678
    lr: float = 1e-3
    # runtime/optim.py (RUNTIME-I20): "adamw" (torch's) or "adamw-host" (the moments in
    # host memory, single-rounding ops). A recipe pin; the reference builds the same.
    optimizer: str = "adamw"
    device: str = "cpu"  # cpu | cuda | mps
    # Heterogeneous fleets: per-slot device, devices[stage][peer]. None = all
    # on `device`. A mixed map trains losslessly but is bit-comparable only to
    # a run on the same map, which is why it is part of the spec.
    devices: list | None = None
    # arch "hf": device_classes[stage][peer] (hfm.device_class) of every slot,
    # pinned (RUNTIME-I15's hf clause): a resume onto another class is refused.
    device_classes: list[list[str]] | None = None
    dataset_path: str | None = None  # pretokenized token file; None = synthetic
    # sha256 of the dataset manifest the job was planned with (model/provision.py): it pins the
    # tokens through the stream's and every shard's digest, which the path alone does not.
    dataset_sha256: str | None = None
    data_shards: int = 1  # recipe-declared mb -> logical shard mapping
    # Where a data-reading slot fetches the manifest and shards it lacks: a directory or an http(s)
    # URL holding the prepared files. Not a pin: every fetched byte is checked against dataset_sha256.
    dataset_source: str | None = None
    steps: int = 8
    microbatches: int = 8  # M per optimizer step
    mb_size: int = 4
    # Keep only the boundary from F to B and rebuild the graph in B: memory
    # for in-flight depth, at the price of one more forward.
    recompute: bool = False
    # RUNTIME-I25: per stage, whether it recomputes; None follows `recompute`. A stage whose order holds one
    # microbatch between its F and its B (always the last) keeps F's graph instead: B's rebuild would hold the same
    # graph for that one microbatch, so it saves a forward at no memory. Builtin models only: a builtin forward
    # yields the same bits with or without its graph (the recompute arms of the bitwise suite), an arch "hf" one
    # need not (mamba's dt_proj). Not a pin, for that reason.
    recompute_stages: list | None = None

    # ------------------------------------------------------------- checkpoints
    ckpt_every: int = 0  # save stage checkpoints every N steps; 0 = off
    # Resume from this slot's own shard in out_dir. The shard's recipe pins
    # (pins()) must equal this spec's; ownership and schedule may change. A
    # missing shard is fatal: a rank at step 0 while its peers resume at N
    # deadlocks the pipeline.
    resume: bool = False
    # The checkpoint step this resume continues from (a restored wave); None
    # = the slot's newest shard. Only meaningful with resume.
    resume_step: int | None = None

    # ------------------------------------------------------------- topology
    stages: int = 2
    stage_widths: list = None  # type: ignore[assignment]  # peers per stage; finalize()
    partition: list = None  # type: ignore[assignment]  # per stage [start, end); finalize()
    # Wide stages: relative throughput of each peer, peer_weights[stage][peer].
    # Microbatches are owned by peers in proportion to it (interleaved weighted
    # round robin, SWARM's routing rule, made static so the schedule can plan
    # it); mb_owner[stage][mb] = peer is derived by finalize().
    peer_weights: list | None = None
    mb_owner: list = None  # type: ignore[assignment]  # finalize() fills
    # Wide stages exchange ZeRO-1 shard gradients per microbatch during the
    # backward tail instead of once at the step end.
    stream_grads: bool = False

    # ------------------------------------------------------------- schedule
    # schedule_order[stage][peer] = [[op, mb], ...] for one step, op in F|B|W,
    # covering exactly the microbatches that peer owns.
    schedule_order: list = None  # type: ignore[assignment]  # finalize() solves
    # While blocked on a late input, run a W the schedule places later if it
    # is already runnable: never violates a dependency, absorbs WAN jitter and
    # cost-estimate error a static schedule cannot.
    schedule_fill: bool = True

    # ------------------------------------------------------------- transport
    port_base: int = 29000
    host: str = "127.0.0.1"
    # Parallel TCP streams per chain hop, for long fat paths where one flow's
    # window cannot fill the pipe; fragments reassemble byte-exactly.
    p2p_streams: int = 1
    # Real-WAN endpoints[stage][peer] = [host, dial_port] or [host, dial_port,
    # listen_port] (NAT / port maps). Overrides host/port_base; workers bind
    # 0.0.0.0.
    endpoints: list | None = None
    # Peer ids (transport/identity.py) allowed on chain hops. None = any
    # authenticated identity.
    trusted_peers: list | None = None
    # slot_peers[stage][peer]: the peer id (32 lowercase hex) expected on that
    # slot. Shape equals stage_widths. None = not pinned (local runs).
    slot_peers: list | None = None
    # Bound on dialing a neighbour that is not up yet, seconds; 0 = forever.
    dial_timeout_s: float = 0.0
    wire_codec: str = "raw"  # raw | fp8_planes | planes (production)
    adapt_budget_ms: float = 0.0  # adaptive microbatching budget; 0 = off
    frag_min_bytes: int = 4096  # fragmentation floor
    # Per-hop WAN emulation in the transport (links[r] joins stage r and r+1).
    links: list = field(default_factory=list)

    # ------------------------------------------------------------- exact sharded stages (ADR-0006)
    # None of these is a recipe pin: they move bytes, never a bit (BA-53).
    # "fp32": exchange gradients widened, at the step end (today). "native": W
    # streams each gradient in the dtype it was produced in, widened at the owner.
    wire: str = "fp32"
    # "zero1": peer k of a wide stage owns params[k::w]. "pieces": peer k owns
    # piece k of every trainable tensor (runtime/pieces.py); needs wire "native".
    shard: str = "zero1"
    # Per stage, the order W emits its deferred gradients (wgrad.FLUSH_ORDERS);
    # None = "store" everywhere (finalize fills it).
    emission_order: list | None = None
    # None, or per stage None or {"<layer index in chunk>.<param name>":
    # [b_0 = 0, ..., b_w = numel]} with every b on pieces.quantum(shape).
    piece_bounds: list | None = None
    # "link": each connection paced alone (today). "shared": every connection of
    # a peer shares its egress and ingress budget, and every connection between
    # two sites the uplink budget of that direction (transport/shim.py).
    pacing: str = "link"
    peer_egress_mbps: list | None = None  # [stage][peer]
    peer_ingress_mbps: list | None = None  # [stage][peer]
    sites: list | None = None  # [stage][peer] site name
    uplink_mbps: dict | None = None  # {"<site>-><site>": mbps}

    out_dir: str = "runs"
    tag: str = "run"

    # ------------------------------------------------------------- derived
    def n_virtual(self) -> int:
        return self.stages

    def default_partition(self):
        """Compute-balanced contiguous cut of [Embed, Block*n, Head]."""
        from wantrain.scheduler.partition import proportional_partition

        return proportional_partition(self.model, [1.0] * self.stages)

    def owned(self, rank: int, peer: int) -> list:
        """Microbatches slot (rank, peer) computes, ascending."""
        return [mb for mb, p in enumerate(self.mb_owner[rank]) if p == peer]

    def finalize(self):
        if self.partition is None:
            self.partition = self.default_partition()
        if not self.links:
            self.links = [asdict(LinkCfg()) for _ in range(self.stages)]
        if self.stage_widths is None:
            self.stage_widths = [1] * self.stages
        assert len(self.stage_widths) == self.stages
        assert len(self.partition) == self.stages, "one layer range per stage"
        assert all(a < b for a, b in self.partition), "empty stage"
        if self.devices is not None:
            assert len(self.devices) == self.stages, "devices must list every stage"
            for r, row in enumerate(self.devices):
                assert len(row) == self.stage_widths[r], f"devices[{r}] must list every peer"
        self._check_one_class()
        from wantrain.runtime.optim import OPTIMIZERS

        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer {self.optimizer!r} is not one of {OPTIMIZERS}")
        if self.optimizer != "adamw" and self.shard == "pieces":
            # pieces (RUNTIME-I12) keeps its own per-unit AdamW over owned pieces.
            raise ValueError(f"optimizer {self.optimizer!r} does not combine with shard 'pieces'")
        if self.recompute_stages is not None:
            if len(self.recompute_stages) != self.stages:
                raise ValueError("recompute_stages must name every stage")
            if (
                self.model.arch == "hf"
                and list(self.recompute_stages) != [self.recompute] * self.stages
            ):
                raise ValueError(
                    "recompute_stages: an arch 'hf' run recomputes on every stage or none"
                )
            self.recompute_stages = [bool(x) for x in self.recompute_stages]
        if self.peer_weights is None:
            self.peer_weights = [[1.0] * w for w in self.stage_widths]
        assert [len(w) for w in self.peer_weights] == list(self.stage_widths)
        if self.mb_owner is None:
            from wantrain.scheduler.zero_bubble import iwrr

            self.mb_owner = [iwrr(self.microbatches, w) for w in self.peer_weights]
        assert len(self.mb_owner) == self.stages
        for r, row in enumerate(self.mb_owner):
            assert len(row) == self.microbatches, f"mb_owner[{r}] must cover every microbatch"
            assert all(0 <= p < self.stage_widths[r] for p in row)
        if self.schedule_order is None:
            from wantrain.scheduler.zero_bubble import default_schedule

            self.schedule_order = default_schedule(self)
        self._check_schedule_order()
        if self.slot_peers is not None:
            assert [len(row) for row in self.slot_peers] == list(self.stage_widths), (
                "slot_peers must have the shape of stage_widths"
            )
            for r, row in enumerate(self.slot_peers):
                for p, pid in enumerate(row):
                    assert isinstance(pid, str) and len(pid) == 32 and set(pid) <= _HEX, (
                        f"slot_peers[{r}][{p}] must be 32 lowercase hex characters"
                    )
        if self.resume_step is not None:
            assert type(self.resume_step) is int and self.resume_step >= 0, (
                "resume_step must be a non-negative int"
            )
            assert self.resume, "resume_step needs resume"
        if self.emission_order is None:
            self.emission_order = ["store"] * self.stages
        self._check_tied_split()
        self._check_m2b()
        return self

    def _tied_stages(self) -> tuple[int, int] | None:
        """(stage of the Embed, stage of the Head) when tie_embeddings puts the
        table's two uses on different stages, else None."""
        if not self.model.tie_embeddings:
            return None
        n_layers = self.partition[-1][1]
        stage_of = {layer: r for r, (a, b) in enumerate(self.partition) for layer in range(a, b)}
        lo, hi = stage_of[0], stage_of[n_layers - 1]
        return None if lo == hi else (lo, hi)

    def _tied_rolled_back(self) -> str | None:
        """The first rolled-back piece whose target refuses a split tied hf
        table (model/rollback.py, read at call time), or None."""
        from wantrain.model import rollback

        for piece in _TIED_PIECES:
            if rollback.target(piece) is not None:
                return piece
        return None

    def tied_relay(self) -> bool:
        """Whether this plan relays a tied table's rows in TIED frames
        (RUNTIME-I19): an hf model whose tied Embed and Head lie on different
        stages, the relay not rolled back."""
        if self.model.arch != "hf" or self._tied_stages() is None:
            return False
        return self._tied_rolled_back() is None

    def _check_tied_split(self):
        """A tied table split across stages (BA-446, BA-447, BA-419). Every arch
        but hf trains a copy per stage, silently wrong: refused. An hf plan
        relays the rows (RUNTIME-I19), which is specified only along a chain of
        width-1 stages from the Embed to the Head, in fp32 boundaries."""
        split = self._tied_stages()
        if split is None:
            return
        lo, hi = split
        n_layers = self.partition[-1][1]
        if self.model.arch != "hf":
            raise ValueError(
                f"tie_embeddings with the Embed and the Head on different stages is refused for "
                f"arch {self.model.arch!r}: the tied parameter layers.0.tok.weight "
                f"(layers.{n_layers - 1}.out.weight) lies on stages {lo} and {hi}, and each "
                f"stage would train its own copy"
            )
        for r in range(lo, hi + 1):
            if self.stage_widths[r] > 1:
                raise ValueError(
                    f"a tied table split across stages {lo} and {hi} needs width 1 on every "
                    f"stage from the Embed to the Head: stage {r} has width "
                    f"{self.stage_widths[r]} (the TIED relay through a wide stage is not "
                    f"specified, ADR-0009 R14, BA-447)"
                )
        if self.model.boundary_dtype != "fp32":
            raise ValueError(
                f"a tied table split across stages {lo} and {hi} needs boundary_dtype 'fp32', "
                f"got {self.model.boundary_dtype!r}"
            )
        piece = self._tied_rolled_back()
        if piece is not None:
            raise ValueError(
                f"a tied table split across stages {lo} and {hi} is refused: piece {piece!r} "
                f"is rolled back to 'tied-split-refused'"
            )

    def _check_m2b(self):
        """The M2b fields (ADR-0006), refused with the fields named. The frozen
        finalize contract (HANDOUT-M2b I-7) accepts every planner output: shard
        "pieces" with wire "native" and stream_grads, wire "native" with
        stream_grads, piece_bounds None or on quanta, one device type per
        pieces stage, a tied Embed and Head on one stage."""
        from wantrain.model.wgrad import FLUSH_ORDERS

        for name, value, allowed in (
            ("wire", self.wire, WIRES),
            ("shard", self.shard, SHARDS),
            ("pacing", self.pacing, PACINGS),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed}")
        eo = self.emission_order
        if not isinstance(eo, list) or len(eo) != self.stages:
            raise ValueError(f"emission_order must list one order per stage, got {eo!r}")
        for r, o in enumerate(eo):
            if o not in FLUSH_ORDERS:
                raise ValueError(f"emission_order[{r}]={o!r} is not one of {FLUSH_ORDERS}")
        if self.shard == "pieces" and self.wire != "native":
            raise ValueError(
                f"shard='pieces' needs wire='native' (wire={self.wire!r}): pieces ship each "
                f"gradient in its production dtype"
            )
        if (
            self.wire == "native"
            and not self.stream_grads
            and any(w > 1 for w in self.stage_widths)
        ):
            raise ValueError(
                "wire='native' with stream_grads=False on a wide stage is refused: the batch "
                "exchange holds every own contribution to the step end (BA-6)"
            )
        if self.shard == "pieces":
            self._check_pieces_class()
            self._check_pieces_tied()
        if self.piece_bounds is not None:
            self._check_piece_bounds()
        if self.pacing == "shared":
            self._check_shared_pacing()

    def _check_one_class(self):
        """CONTROL-I25 (RUNTIME-I15): a devices map of more than one device
        type, or a device_classes map of more than one class, is refused."""
        for name, rows, kind in (
            ("devices", self.devices, lambda d: str(d).split(":")[0]),
            ("device_classes", self.device_classes, str),
        ):
            if rows is None:
                continue
            slots = [
                (f"{name}[{r}][{p}]", kind(x))
                for r, row in enumerate(rows)
                for p, x in enumerate(row)
            ]
            mixed = mixed_fleet(slots)
            if mixed is not None:
                raise ValueError(mixed)

    def _check_pieces_class(self):
        if self.devices is None:
            return
        for r, row in enumerate(self.devices):
            kinds = sorted({d.split(":")[0] for d in row})
            if len(kinds) > 1:
                raise ValueError(
                    f"shard='pieces' needs one device type per stage: stage {r} mixes "
                    f"{kinds} (devices[{r}]={row})"
                )

    def _check_pieces_tied(self):
        if not self.model.tie_embeddings:
            return
        n_layers = self.partition[-1][1]
        stage_of = {layer: r for r, (a, b) in enumerate(self.partition) for layer in range(a, b)}
        if stage_of.get(0) != stage_of.get(n_layers - 1):
            raise ValueError(
                f"shard='pieces' with tie_embeddings needs the Embed and the Head on one stage: "
                f"the tied parameter layers.0.tok.weight (layers.{n_layers - 1}.out.weight) "
                f"lies on stages {stage_of.get(0)} and {stage_of.get(n_layers - 1)}"
            )

    def _check_piece_bounds(self):
        from wantrain.runtime import pieces

        pb = self.piece_bounds
        if not isinstance(pb, list) or len(pb) != self.stages:
            raise ValueError(f"piece_bounds must list one entry per stage, got {pb!r}")
        if all(x is None for x in pb):
            return
        shapes = self.stage_param_shapes()
        for r, bounds in enumerate(pb):
            if bounds is None:
                continue
            if not isinstance(bounds, dict):
                raise ValueError(f"piece_bounds[{r}] must be None or a dict, got {bounds!r}")
            try:
                pieces.piece_layout(shapes[r], self.stage_widths[r], bounds)
            except ValueError as e:
                raise ValueError(f"piece_bounds[{r}]: {e}") from None

    def stage_param_shapes(self) -> list:
        """Per stage, {"<layer index in chunk>.<param name>": shape} of its
        trainable parameters, in named_parameters order. Builds the model on
        the meta device under a forked RNG, so the global torch RNG and memory
        are untouched. arch "hf" reads the builder's census instead (I-10) and
        never imports hfm or transformers: finalize runs in the coordinator and
        the planner (BA-419, BA-441). An MoE model's selection biases (buffers)
        follow its parameters: pieces cut them like any small tensor (BA-455)."""
        if self.model.arch == "hf":
            return self._census_param_shapes()
        import torch

        from wantrain.model.gpt import build_layers

        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            layers = build_layers(self.model, self.init_seed)
        out = []
        for a, b in self.partition:
            chunk = torch.nn.ModuleList(list(layers)[a:b])
            shapes = {n: tuple(p.shape) for n, p in chunk.named_parameters() if p.requires_grad}
            if getattr(self.model, "moe_experts", 0):
                shapes.update(
                    {n: tuple(t.shape) for n, t in chunk.named_buffers() if _is_sel_bias(n)}
                )
            out.append(shapes)
        return out

    def _census_param_shapes(self) -> list:
        census = self.model.hf_census
        if not isinstance(census, list):
            raise ValueError("arch 'hf' needs model.hf_census (the builder's unit census, I-10)")
        n_units = self.partition[-1][1]
        if len(census) != n_units:
            raise ValueError(f"model.hf_census lists {len(census)} units, the partition {n_units}")
        out = []
        for a, b in self.partition:
            shapes: dict = {}
            for u in range(a, b):
                unit = census[u]
                rows = unit.get("param_shapes") if isinstance(unit, dict) else None
                if not isinstance(rows, list):
                    raise ValueError(f"model.hf_census[{u}] has no param_shapes")
                for row in rows:
                    if (
                        not isinstance(row, (list, tuple))
                        or len(row) != 2
                        or not isinstance(row[0], str)
                        or not isinstance(row[1], (list, tuple))
                        or any(type(d) is not int or d < 0 for d in row[1])
                    ):
                        raise ValueError(f"model.hf_census[{u}].param_shapes holds {row!r}")
                    shapes[f"{u - a}.{row[0]}"] = tuple(row[1])
            out.append(shapes)
        return out

    def _check_shared_pacing(self):
        def grid(name):
            v = getattr(self, name)
            if not isinstance(v, list) or [len(row) for row in v] != list(self.stage_widths):
                raise ValueError(
                    f"pacing='shared' needs {name}[stage][peer] for every slot, got {v!r}"
                )
            return v

        for name in ("peer_egress_mbps", "peer_ingress_mbps"):
            for r, row in enumerate(grid(name)):
                for p, x in enumerate(row):
                    if isinstance(x, bool) or not isinstance(x, (int, float)) or not x > 0:
                        raise ValueError(f"pacing='shared': {name}[{r}][{p}]={x!r} is not > 0")
        sites = grid("sites")
        for r, row in enumerate(sites):
            for p, s in enumerate(row):
                if not isinstance(s, str) or not s or "->" in s:
                    raise ValueError(f"pacing='shared': sites[{r}][{p}]={s!r} is not a site name")
        up = self.uplink_mbps if isinstance(self.uplink_mbps, dict) else {}
        for a, b in sorted(self.site_pairs()):
            x = up.get(f"{a}->{b}")
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not x > 0:
                raise ValueError(
                    f"pacing='shared': uplink_mbps[{a + '->' + b!r}] is missing or not > 0"
                )

    def budget_paths(self) -> dict:
        """pacing "shared": the budget file of every peer egress ("egress", r,
        p), peer ingress ("ingress", r, p) and site pair direction ("uplink",
        from, to), under out_dir/budgets/<tag>/ (BA-55). The local launcher
        creates them fresh; every slot opens them by these paths."""
        import os

        root = os.path.join(self.out_dir, "budgets", self.tag)
        sites = sorted({s for row in (self.sites or []) for s in row})
        out: dict = {}
        for r, w in enumerate(self.stage_widths):
            for p in range(w):
                out[("egress", r, p)] = os.path.join(root, f"egress_r{r}p{p}")
                out[("ingress", r, p)] = os.path.join(root, f"ingress_r{r}p{p}")
        for a, b in sorted(self.site_pairs()):
            out[("uplink", a, b)] = os.path.join(root, f"uplink_{sites.index(a)}_{sites.index(b)}")
        return out

    def site_pairs(self) -> set:
        """Ordered (from, to) site pairs of every connection between two sites."""
        pairs: set = set()
        sites = self.sites or []
        for r, w in enumerate(self.stage_widths):
            for p in range(w):
                here = sites[r][p]
                peers = [(r, q) for q in range(w) if q != p]
                if r + 1 < self.stages:
                    peers += [(r + 1, q) for q in range(self.stage_widths[r + 1])]
                if r > 0:
                    peers += [(r - 1, q) for q in range(self.stage_widths[r - 1])]
                for st, q in peers:
                    there = sites[st][q]
                    if there != here:
                        pairs.add((here, there))
        return pairs

    def pins(self) -> dict:
        """The recipe pins a checkpoint records and a resume must match: every
        field a shard's bytes depend on. JSON-normalized (tuples are lists),
        so a stored copy compares equal to a fresh one.

        Not pins: peer_weights, mb_owner, schedule_order, endpoints, devices,
        slot_peers, trusted_peers, resume_step, out_dir, dataset_source (who computes what
        never moves a bit, the fold order is fixed) and steps (no update reads
        the total step count; a resume may extend it).

        device_types pins where: per stage, the sorted distinct torch device
        types of its slots. A kernel's bits depend on the device type (and
        sdpa_backend "auto" resolves by it), so devices may change only within
        those types.

        zero1_widths binds whole-tensor ZeRO-1 shards only: peer k of a width-w
        stage holds the optimizer state of params[k::w], which depends on w.
        RUNTIME-I12 logical checkpoints are width-free and M2b drops this pin
        for them.
        """
        pins = {
            "model": _model_pins(self.model),
            "init_seed": self.init_seed,
            "data_seed": self.data_seed,
            "dataset_path": self.dataset_path,
            "data_shards": self.data_shards,
            "microbatches": self.microbatches,
            "mb_size": self.mb_size,
            "lr": self.lr,
            "adamw": ADAMW,
            "wire_codec": self.wire_codec,
            "recompute": self.recompute,
            "partition": self.partition,
            "device_types": [
                sorted({self.device_for(r, p).split(":")[0] for p in range(w)})
                for r, w in enumerate(self.stage_widths)
            ],
            "zero1_widths": self.stage_widths,
        }
        if self.model.arch == "hf":
            pins["device_classes"] = self.device_classes
        # Only when not torch's AdamW, so every existing checkpoint keeps its pins.
        if self.optimizer != "adamw":
            pins["optimizer"] = self.optimizer
        # Only when set: a synthetic run, and a checkpoint written before the corpus was pinned by
        # content, keep the pins they had (an absent key differs from a present None).
        if self.dataset_sha256:
            pins["dataset_sha256"] = self.dataset_sha256
        return json.loads(json.dumps(pins))

    def reschedule(self):
        """Re-derive microbatch ownership and the analytic schedule after the
        topology, the peer weights or the partition changed."""
        self.mb_owner = None  # type: ignore[assignment]
        self.schedule_order = None  # type: ignore[assignment]
        return self.finalize()

    def _check_schedule_order(self):
        assert len(self.schedule_order) == self.stages, "schedule_order needs every stage"
        for r, per_peer in enumerate(self.schedule_order):
            assert len(per_peer) == self.stage_widths[r], f"schedule_order[{r}] needs every peer"
            for p, order in enumerate(per_peer):
                want = self.owned(r, p)
                seen: dict = {op: [] for op in OPS}
                for op, mb in order:
                    assert op in OPS, f"unknown op {op!r}"
                    seen[op].append(int(mb))
                for op in OPS:
                    assert sorted(seen[op]) == want, (
                        f"slot ({r}, {p}): {op} must cover exactly its microbatches"
                    )

    def recomputes(self, rank: int) -> bool:
        """Whether stage `rank` recomputes its activations in B (RUNTIME-I25)."""
        if self.recompute_stages is None:
            return bool(self.recompute)
        return bool(self.recompute_stages[rank])

    def device_for(self, rank: int, peer: int = 0) -> str:
        """The torch device string this slot runs on."""
        if self.devices is not None:
            return self.devices[rank][peer]
        return self.device

    def to_json(self) -> str:
        d = asdict(self)
        d["model"] = asdict(self.model)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> JobSpec:
        d = json.loads(s)
        d["model"] = GPTConfig(**d["model"])
        return JobSpec(**d)

    @staticmethod
    def load(path: str) -> JobSpec:
        with open(path) as f:
            return JobSpec.from_json(f.read())


def _model_pins(model: GPTConfig) -> dict:
    """asdict(model) without the M4 fields a model does not use (BA-465): the
    hf_* fields of a non-hf model and the moe_* fields of a dense one, so a
    dense model's pins are the pre-M4 bytes."""
    d = asdict(model)
    if model.arch != "hf":
        for k in _HF_FIELDS:
            d.pop(k, None)
    if not d.get("moe_experts"):
        for k in [k for k in d if k.startswith("moe_")]:
            d.pop(k)
    return d


def _is_sel_bias(name: str) -> bool:
    return name == "sel_bias" or name.endswith(".sel_bias")


def first_pin_difference(want: dict, have: dict) -> str | None:
    """The dotted name of the first pin that differs between two pins()
    dicts (want's key order first), or None when they are equal."""
    for k in list(want) + [k for k in have if k not in want]:
        a, b = want.get(k), have.get(k)
        if a == b and (k in want) == (k in have):
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            return f"{k}.{first_pin_difference(a, b)}"
        return k
    return None


def vs_rank(spec: JobSpec, vs: int) -> int:
    """Rank hosting stage vs (one stage per rank)."""
    return vs
