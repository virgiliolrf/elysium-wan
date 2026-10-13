# SPEC-runtime: stage execution, the job spec and the oracle

**Workstream:** `runtime`. **Owned paths:**
- `src/wantrain/runtime/`, `src/wantrain/launch/`, `src/wantrain/validation/`, `src/wantrain/worker/stage_main.py`,
  `src/wantrain/__init__.py`;
- `tests/integration/test_engine_bitwise.py`, `tests/unit/test_user_model_precision.py`, `tests/unit/test_runtime_spec.py`.

**Status:** describes what exists on `main` (M0) plus the ordered tasks for later milestones. Design of record:
[DESIGN.md](DESIGN.md).

## 1. Scope

The runtime executes one slot `(rank, peer)` of a pipeline:
- it connects to its neighbours;
- it runs `schedule_order[rank][peer]` once per step;
- it folds gradients in microbatch order, steps the optimizer, checkpoints, and reports.

It also owns `JobSpec`, the single object that determines a run, and the oracle that every run is diffed against. It
owns no scheduling decision: orders, ownership and partitions arrive in the spec.

## 2. Domain model (normative)

- **Operations.** For each microbatch `mb` a slot owns, there are exactly one `F`, one `B` and one `W`.
  - `F(mb)`: run the slot's layers on the input activation.
    - Stage 0 reads the batch `make_batch(data_seed, step, mb, ...)` instead.
    - With `recompute`, F runs under `no_grad` and keeps only the boundary.
  - `B(mb)`: rebuild the graph if recomputing, then compute `dL/dX` only. Every `SplitLinear` captures `(X, dY)` into the
    weight-gradient store (`model/wgrad.capture`) instead of computing `dW`.
    - Parameters outside split linears (embeddings, norms, tied weights) get their gradients in B.
    - The last stage starts from `loss_fn(logits, targets)`.
    - `dX` is sent upstream at once and the graph is freed.
  - `W(mb)`: `wgrad.flush(store)` computes `dW = dYᵀX` for that microbatch. The result is assembled with B's other
    gradients and stored for the fold.
- **Dependencies.**
  - `F(mb)` on stage `s>0` needs `FWD(mb)` from stage `s-1`.
  - `B(mb)` on stage `s<S-1` needs `BWD(mb)` from stage `s+1`.
  - `W(mb)` needs `B(mb)` on the same slot.
  - (M2b, sharded stages) F of step t+1 on a unit also needs that unit's step-t working copy (RUNTIME-I13).
  - Nothing else. W never waits on the network.
- **Ownership.**
  - Width-1 stages own every microbatch.
  - Wide stages own `mb_owner[rank][mb] == peer`. The owner is static per attempt (IWRR over `peer_weights`); it changes only at a resume (RUNTIME-I15).
  - Forward routing picks the owner in the next stage. Backward routing returns to the slot that sent the forward.
- **Fold.** Gradients for a step are summed in **microbatch-index order**, regardless of arrival order
  (`OrderedAccumulator`).
- **Optimizer.** AdamW.
  - Width 1: local.
  - Wide stage, ZeRO-1: peer `k` owns `params[k::w]`; every peer ships per-microbatch shard gradients to the owner. The
    owner sums all M in index order, updates, and broadcasts the new shard. With `stream_grads`, the shipping overlaps the
    backward tail. (M4, ADR-0008) For a stage with MoE selection biases the ownership list is its trainable names in
    order, then its selection biases in layer order (a dense stage's list is unchanged); RUNTIME-I18 carries them.
- **Execution.** `_run_step` walks the order. When the next op's input is missing and `schedule_fill` is set, the slot
  runs the earliest already-runnable W the order places later. This never violates a dependency, and the numerics are
  unaffected because the fold order is fixed. A W left over at the step end is an error.
- **Checkpoint.** Every `ckpt_every` steps: layer state, optimizer state and the loss history up to that step, plus a
  sidecar with the step.
  - Resume loads the slot's own shard.
  - **RUNTIME-I17.** Every checkpoint of step n is committed as an immutable wave
    `out_dir/waves/<tag>/r<rank>p<peer>/step-<n:08d>/`:
    - its files are written first;
    - `wave.json` (`{"v": 1, "step": n, "files": [{"name", "sha256", "bytes"}]}`) is written last;
    - the directory is renamed into place.

    A committed wave is never written again, and a second commit of the same step writes nothing. After a new commit,
    the slot derives the alias `<tag>_ckpt_rank<r>p<p>.pt` + `.step.json` from the wave (read only by the oracle
    readers; every writer replaces it, none writes into it), prints `WT_WAVE <n>`, and keeps its newest 3 waves.

    Resume reads only the wave of `resume_step` (the newest, in local mode). Before any state is applied, resume
    refuses:
    - a missing wave or `wave.json`;
    - a `wave.json` naming another step;
    - a listed file that is missing or has another sha256 or size;
    - an unlisted file.

    A resume from step n then removes the slot's waves newer than n; a launch that does not resume removes every wave
    of the slot.

    `wave.json` is the wave's sidecar (RUNTIME-I12): M2b lists piece files in it.
  - The state file records the recipe pins: model, seeds, data, microbatches, `mb_size`, `seq_len`, optimizer and
    hyperparameters, precision, wire codec, recompute and partition. It also records `device_types`: per stage, the
    sorted distinct torch device types (`cpu`, `cuda`, `mps`; the part of a device string before `:`) of its slots,
    taken from `devices`, or `device` for every slot when there is no map. A kernel's bits depend on the device type
    (DESIGN §3: bit-comparable only on the same device map; `sdpa_backend: auto` resolves by device type and compute
    dtype, both now pinned), so `devices` may change only within those types. For whole-tensor ZeRO-1 shards it also records
    `zero1_widths` (the stage widths), because `names[k::w]` depends on w. A logical-layout checkpoint (RUNTIME-I12) is
    width-free and is checked through its manifest. (M4, BA-465) The model pin, and `recipe_digest`'s model dict
    (`launch/local.py`), is `asdict(model)` less, for a non-hf model, the `hf_*` fields M4 added (`hf_config`,
    `hf_buffers`, `hf_census`, `hf_code`, `hf_impl`) and, when `moe_experts == 0`, every `moe_*` field; the pre-M4
    `hf_model` and `hf_model_sha256` stay, so dense pins and digests are byte-identical to 3b0d2f8's.
  - `steps` is not a pin: no update depends on the total step count (the loop bound is its only consumer), so a run
    cut at step k is a prefix of the same recipe with more steps, and a resume may extend `steps`. Resume refuses a
    `steps` below the first step it would run (checkpoint step + 1), because the final checkpoint would then label
    that state with an earlier step. A field that makes any step's update depend on the total step count (an LR
    schedule's horizon, for example) is a pin from the change that introduces it.
  - Resume refuses a state file whose pins differ or that has none. `peer_weights`, `mb_owner` and `schedule_order`
    may change at a resume.
- **Shutdown.** After the last step, each sender drains, sends `END` and half-closes. The reader keeps draining until
  every neighbour's `END`. EOF without `END` is fatal.
- **Oracle.** `validation/reference.run_baseline` trains the same recipe on one device, in the same microbatch order,
  with the same fold. `validation/compare` diffs loss sequences as float hex. (M4, ADR-0008) For an MoE model the
  oracle also records each MoE layer's aux values per (step, mb) as hex; the objective's gradient comes from the
  block's own injection (MODEL-I8), never from a summed loss scalar; after each AdamW step it applies the
  selection-bias update from the step's integer counts. (M4, ADR-0009) For `arch="hf"` the oracle runs, with the
  run's loaded parameters and the buffers of the run's buffer record (RUNTIME-I19), eager attention and
  `loss_fn(logits.float(), targets)`: on a map of one device class, the vanilla `ForCausalLM.forward` in one process
  on that class (the datacenter's function, so identity checks the units against HF at full size); on a mixed map
  one node holds, the run's units (`hfm.units`, MODEL-I9) in one process, each on its map device. It folds exactly as
  for every arch: one `autograd.grad` per microbatch over the model's parameters (a tied table's uses summed by
  autograd), then the index order fold. Any other map is refused (MODEL-I9, CONTROL-I24). The fold is unchanged.
  An HF MoE model's oracle carries the same SUBSTITUTE class swap and aux injection as the run (ADR-0008 decision
  11, the HF-MoE seam rule), never HF's own MoE block; the block itself is proved by MODEL-I8.

## 3. Interfaces

**Provides**
- `JobSpec` (dataclass, JSON round trip: `to_json`, `from_json`, `load`).
  - `finalize()` fills `partition`, `links`, `stage_widths`, `peer_weights`, `mb_owner` (IWRR) and `schedule_order`
    (`scheduler.zero_bubble.default_schedule`), then validates.
  - `reschedule()` clears ownership and the order, then finalizes again.
  - `owned(rank, peer) -> list[int]`, `device_for(rank, peer) -> str`.
- `StageRuntime(spec, rank, peer).run() -> None`. Prints `WT_STAGE`, `WT_PHASE`, `WT_STEP <n>`, `WT_METRIC`, `WT_MEM`
  and `WT_RESUME`. `stage_main` prints `WT_DONE <rc>`.
- `OrderedAccumulator(order)`: `.add(mb, tensors, verified=True)` (unverified contributions wait in quarantine), `.release(mb)`,
  `.reject(mb, replacement)` (the replacement, e.g. an honest recompute, takes the slot), `.complete() -> bool`, `.result()`.
- `setup_determinism(seed)`, `float_hex(x)`.
- `run_baseline(spec, tag, ...)`, `compare(baseline_path, candidate_path)`, `run_local(spec)`,
  `run_and_compare(spec)`.

**Consumes**
- model: `build_layers`, `loss_fn`, `wgrad.capture/flush/convert/owned_params`, `precision.spec`, `make_batch`.
- transport: `send_frame`, `recv_frame`, `send_end`, `Reassembler`, `authenticate`, `SecureSocket`, `hardware_aes`,
  `serial.dumps/loads`, `LinkSpec`, `SenderPacer`, `DelayedDelivery`.
- codec: `planes.encode/decode`, `fp8_planes`, `f32_planes`.
- scheduler: `iwrr`, `default_schedule`, `proportional_partition`.

## 4. Invariants

Each one is tested; the covering test carries `covers: <id>`.

- **RUNTIME-I1.** The per-microbatch loss sequence equals the oracle **bit for bit** for every mechanism:
  - 2–4 stage chains, with and without WAN latency and jitter;
  - recompute; fp8 and planes wires;
  - adaptive microbatching; multi-stream hops;
  - wide stages 2-1, 1-2, 2-2, 1-2-1, and IWRR-weighted;
  - streamed ZeRO-1 gradients;
  - strict order without W fill.
- **RUNTIME-I2.** Any valid schedule changes timing, never numerics. Orders solved from paper costs, profiled costs or
  heterogeneous costs all give oracle-identical losses.
- **RUNTIME-I3.** Checkpoint then resume continues the loss sequence bit for bit, including the history before the
  resume point.
- **RUNTIME-I4.** The fold order is the microbatch index, independent of arrival order. A quarantined contribution
  delays the fold and never reorders it. A rejected one never enters it.
- **RUNTIME-I5.** `JobSpec.finalize()` rejects a schedule whose F, B or W does not cover **exactly** the slot's owned
  microbatches, and one with an unknown operation.
- **RUNTIME-I6.** Compute precision means the same for user models as for built-ins. The engine opens the autocast
  region only for user models, built-ins are unchanged, and recompute reproduces the forward exactly.
- **RUNTIME-I7.** A normal shutdown never kills a slower rank: a slot exits only after every neighbour's `END`. This is
  the 3-stage, 20 ms + 40 ms jitter regression from the source.
- **RUNTIME-I9.** Each slot prints one `WT_TELEM` per optimizer step, before `WT_STEP`: per-op F/B/W/OPT count and median
  ms, stall ms, per-hop frames, bytes and wire-busy ms, per inbound hop the count, median and max of each boundary's
  spread (first fragment to whole, the `recv` event), and its own emit cost, computed from monotonic timestamps only
  (no tensor op, no device sync, no RNG draw). The training state stays byte-identical to the reference.
- **RUNTIME-I10.** (DEFERRED-TO: M2b) Sharded stages (`shard: "pieces"`, ADR-0006) and the native wire dtype
  (`wire: "native"`) are byte-identical.
  - The sha256 of the logical training state (weights, AdamW m, v and step, RNG) and every loss hex equal the oracle.
  - This holds at every width w ≥ 1, every piece split, every compute dtype (fp32, bf16, fp16), every IWRR ownership
    (equal or weighted), every zero-bubble order and W fill, both emission orders (MODEL-I7), and with per-unit updates.
  - A working copy carries its step per unit. An op that reads a unit copy of another step fails.
  - A GRADS or PARAMS frame whose dtype differs from the plan's table, whose shape differs from the piece, whose
    microbatch index is outside `[0, M)`, whose step is neither the receiver's current nor its next step, or that
    repeats a (step, mb, unit, sender) is fatal.
- **RUNTIME-I11.** (DEFERRED-TO: M2b) The piece fold and the piece-local cast and AdamW equal the whole-tensor reference
  per element, under the pinned AdamW implementation, at every piece split whose boundaries are multiples of
  `lcm(row length, 128)` elements, including a tensor with fewer quanta than peers. The fold accumulator is an fp32
  copy of microbatch 0's widened piece: never a clone in the wire dtype, never zeros.
- **RUNTIME-I12.** (DEFERRED-TO: M2b) Checkpoints of a sharded stage are in logical layout: per-piece files under a
  `(name, offset, length)` manifest, with a sha256 per chunk.
  - Resume at another width or piece split is byte-identical.
  - A manifest whose pieces do not tile every tensor exactly, or whose chunk hash mismatches, is refused.
  - The manifest is embedded in every piece file and names the step, the wave's width, the slot's peer index and the
    layout; the wave's sidecar names the same step, and the file's sha256 binds both. A resume at any width reads
    exactly the files whose manifests name the wave's step and width, peer indices 0..w−1, and refuses a missing or
    extra index, a file whose manifest is missing or names another step, like a missing shard.
  - Whole-tensor ZeRO-1 checkpoints are read as `names[k::w]`. A logical-layout checkpoint is read by `shard: "zero1"`
    as `names[k::w]` of the logical tensors, byte-identically, so `shard` rolls back.
- **RUNTIME-I13.** (DEFERRED-TO: M2b) The per-unit step loop of a sharded stage is live and bounded.
  - A unit's update (AdamW on the owned piece, the cast, the gather send) runs in the first `_pump` after its fold
    completes, wherever the slot is: between ops or inside any blocked wait, of step t or t+1. Under every valid order,
    W fill, arrival jitter and one peer 2× slow for a whole run, every step completes.
  - An F of step t+1 on a unit waits only for that unit's step-t copy. Checkpoint, re-plan and last steps keep the full
    step barrier.
  - With `M_k` the microbatches peer k owns, its exchange bytes in host memory (gradient pieces held or queued, and
    gathers queued to send or received and awaiting their H2D copy) never exceed
    `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w` bytes, with `b_g·P_s = Σ_u b_g,u·P_u` and
    `b_p·P_s = Σ_u b_p,u·P_u` over the stage's units: one step of gradients, and per unit at most one step of gathers
    out and one in (ADR-0006 `round5.md` §4). Its own pieces waiting on the device for their prefix never exceed
    `Σ_u b_g,u·M_k·P_u/w` bytes (one step; `b_g·(M/w)·P_s/w` under equal weights).
  - Each per-destination send queue serves need-first: forward unit index, then gathers before gradients, then step
    and microbatch; one chunk of at most 16 MiB at a time, never preempted inside a chunk.
  - The slot's compute thread never performs a blocking host↔device copy of exchange bytes: D2H and H2D run on a copy
    stream in pinned chunks of at most 16 MiB, ordered by events.
- **RUNTIME-I15.** A resume whose spec changes only `peer_weights`, `mb_owner`, `schedule_order`, `endpoints`,
  `devices` (keeping the `device_types` pin), `slot_peers`, `resume_step` or (upward) `steps` continues
  byte-identically to the oracle of the resumed spec's recipe: losses, weights, optimizer state and RNG. A resume whose
  pins differ (a slot moved to another device type differs in `device_types`), or whose `steps` is below the
  checkpoint step + 1, is refused before any state is applied, naming the field.
  - Scope of the identity claim: the oracle runs on `device`, so the claim covers runs whose every slot has `device`'s
    type. (The human, 2026-10-11; CONTROL-I25) Only one-class maps run, for every arch: `JobSpec.finalize` refuses a
    `devices` map spanning more than one device type (`MIXED-FLEET`), so a mixed map neither launches nor resumes.
  - A same-type move between GPU models (one CUDA card for another) is not refused: its identity rests on the product
    rule's ban on architecture-dependent kernels and is real-hardware validation, N/A while `gpu` is false. The run's
    `_env.json` fingerprint records the GPU name, so such a move stays visible. For `arch="hf"` a move to another
    device class (GPU model, or a CPU's ATen capability) is refused, and every resume re-checks decision 8 on the map
    it builds (CONTROL-I24; ADR-0009 decision 8).
- **RUNTIME-I16.** Every data-plane socket, chain or mesh, is authenticated (TRANSPORT-I5) before any frame and then
  sealed (TRANSPORT-I7).

  An accepted socket is refused, logged (`WT_P2P_REFUSED`) and closed, while accepting continues, when:
  - its key fails, or it sends nothing within the handshake timeout;
  - its id is not `slot_peers[st][pe]` (on the chain, `slot_peers[rank−1][0]`);
  - `st ∉ {rank−1, rank}`;
  - `pe` is out of range, or for intra `pe ≤` own peer;
  - `(st, pe)` is already bound.

  A refused socket never consumes a legitimate peer's slot or stream.

  A coordinator-run spec (endpoints set) without `slot_peers` is refused. Each legitimate socket yields exactly one
  `p2p_auth` and one `p2p_connected` event. Each slot reports its seal and open time (`WT_AEAD`).
- **RUNTIME-I17.** Immutable checkpoint waves and the resume that reads only them: stated in §2 Checkpoint.
- **RUNTIME-I8.** (DEFERRED-TO: M3) ZB-V: a rank hosting two virtual chunks in V order (chunk 0 forward down, chunk 1 forward up) keeps
  RUNTIME-I1 and RUNTIME-I2.
- **RUNTIME-I14.** (DEFERRED-TO: M4; ADR-0007) A slot materializes only its slice, and proves its initial bytes.
  - With `build: "slice"`, a slot at step 0, or relaunched before its run's first wave, builds `build_slice` (MODEL-I6)
    with its device as home and `JobSpec.init_record`. Its host memory during the build stays within one working
    layer and a held tied table, `4·(P_layer,max + P_held)` bytes over the interpreter's baseline (MODEL-I6), plus its
    slice, `4·P_s`, when its device is the CPU. A slot of
    a sharded stage (`shard: "pieces"`) trims each home tensor when it is finalized: fp32 only for its own pieces, and
    its step-0 working copy cast from the built values, the bytes a gather would bring.
  - The slot process calls `setup_determinism` before its first kernel dispatch (today in `StageRuntime.__init__`,
    `stage.py:130`); a kernel dispatched earlier is a defect. Before its first draw the
    slot compares `sampler_fingerprint()` with the record's and raises `INIT-CLASS`, naming both fingerprints and both
    diagnostic classes, on a difference. A launch with `build: "slice"` and no record, or with a record whose binding
    is not that of the spec's (cfg, seed, program id, code id; MODEL-I6), raises `INIT-RECORD`. `INIT-LEDGER` and `INIT-DIGEST`
    (MODEL-I6) are fatal for the slot and reported to the coordinator; the report says whether an `INIT-LEDGER` is at
    the anchor (the record's fault, CONTROL-I19) or at a transition.
  - The record is validated at the boundary before use: its size, before parsing, at most `(n_phase·(L+2) + 1)·6760 +
    Σ_t (len(name_t) + 80) + 65536` bytes over `skeleton(cfg)`'s tensor names (1.23 MB at 70B, 3.05 MB at 405B with
    LoRA `all`; ADR-0007 `cost_r5.py`); JSON; every state exactly 5056 bytes once base64-decoded; every tensor name in
    `skeleton(cfg)`; the binding checked before any draw. The identity manifest records the record's
    sha256, binding and fingerprint.
  - `run_local` makes the record with `init_record` in a fresh process when the spec has none (one host, one class).
    `validation/reference.py` builds with `build_layers` and, when the recipe carries a record, checks before step 1
    the record's binding and fingerprint, `get_rng_state()` at every (phase, layer) boundary of its phase-major walk
    against the ledger, and every tensor's sha256 against the record.
  - A resume, at any width or piece split (RUNTIME-I12), never draws from the generator: the slot takes its slice of
    `skeleton(cfg)`, `to_empty` on its device, re-tied from the skeleton's sharing map (MODEL-I6), and loads every
    tensor and the RNG state from the wave (for `arch="hf"` also the buffers, checked against the pin, RUNTIME-I19). Under `arch="user"`, which has no skeleton, a resume keeps today's full
    build, then loads every tensor and the RNG state from the wave (the build's draws are overwritten). A wave,
    whole-tensor or logical, that lacks the RNG state or any tensor of the slice is refused, naming the file. The slot
    needs no record and compares no fingerprint on resume; which node may heal a slot is CONTROL-I19's. Its host
    memory is the wave as loaded (`torch.load` to the CPU: fp32 weights and the optimizer's `m` and `v`, `12·P_s`),
    plus the slice when its device is the CPU (SCHEDULER-I18 charges it).
  - `build: "full"` keeps today's full build and slice, with no record; `arch="user"` always uses it. The draw-free
    resume (for builtin, general and hf) and the refusal of a wave without the RNG state hold under both values: `build`
    does not roll them back; they roll back by code revert, with no `resume` field (ADR-0007 R8, decision 5).

- **RUNTIME-I18.** (DEFERRED-TO: M4; ADR-0008) MoE models are byte-identical, for every recipe row of ADR-0008.
  - The sha256 of the training state (weights, AdamW m, v and step, RNG, every MoE layer's selection bias), every
    loss hex and every aux hex per (step, mb, layer) equal the oracle's at every pipeline depth and cut, every
    zero-bubble order and W fill, with recompute, at every ZeRO-1 width and (once M2b is merged) every piece width, in
    fp32 and bf16, under any routing skew, including experts routed no token in a microbatch or a whole step.
  - The selection-bias update of step t uses the int64 counts of every microbatch of step t, applied by the bias's
    owner after the step's AdamW. A selection bias is a stage tensor updated by that rule instead of AdamW; each
    microbatch's count vector (E values per MoE layer, summing to `T_mb·k`) is its per-microbatch input and travels
    where a trainable tensor's gradient travels:
    - width 1: local;
    - `shard: "zero1"`: the bias's owner is given by §2's ownership list; each peer sends, with a microbatch's shard
      gradients to owner `k`, the count vectors of the biases `k` owns; `k` folds them by integer add in microbatch
      order, updates, and broadcasts the bias with its new shard; shard checkpoints and the replica-agreement check
      cover it;
    - `shard: "pieces"`: the bias is a tensor of its block's small-tensor unit, cut like any of its tensors; the count
      vector is cut at the same boundaries and rides the unit's GRADS frame in an int64 lane; the piece owner folds,
      updates and gathers the piece with the unit's working copy; RUNTIME-I12's logical checkpoint and manifest
      include it.
    Step t+1's F of that layer waits for the new bias (zero1's broadcast, or the unit's gather). A count lane whose
    dtype is not int64, whose shape differs from the piece's (or E under zero1), whose (step, mb, sender) breaks the
    GRADS rules or that holds a negative count is fatal; under zero1 a vector whose sum differs from `T_mb·k` is fatal;
    a count vector missing at the owner for any (microbatch, layer) is fatal like a missing gradient.
  - A trainable parameter without a gradient in B/W stays fatal (`stage.py`'s guard), as the oracle's
    `autograd.grad` without `allow_unused` does.
  - The slot owning a microbatch records each of its MoE layers' aux hex and integer counts from F, once per
    (step, mb, layer), and keeps the history in its checkpoint like the loss history. B's rebuild recomputes them and
    a difference is fatal (`MOE-REPLAY`, naming step, microbatch and layer). The identity manifest gains an `aux`
    section, `[step, mb, layer, hex]` sorted; a (step, mb, layer) recorded twice or missing fails the comparison.
- **RUNTIME-I19.** (DEFERRED-TO: M4; ADR-0009) HF units and tied tables are byte-identical to the oracle.
  - The sha256 of the training state and every loss hex equal the oracle's (§2 Oracle) for every gate-admitted
    family, at every cut, zero-bubble order, recompute, ZeRO-1 width, piece width (once M2b is merged) and resume,
    with each slot's units from `hfm.units` (MODEL-I9), every slot's fingerprint admitted (CONTROL-I24), and the
    determinism asserts at stage start:
    deterministic algorithms with `warn_only` false, fp32 matmul precision `highest`, `_attn_implementation == "eager"`,
    every dropout probability 0, HF gradient checkpointing off.
  - The buffer record (every skeleton buffer's bytes and a sha256 per tensor) is built once, at the first launch
    (SPEC-control §2.1), and pinned in the JobSpec and the state file. Every slot writes it as a file listed in each
    wave's `wave.json` (RUNTIME-I17's format unchanged). A launch, resume, heal, restart or rebalance loads the buffers
    from the JobSpec or the wave and refuses, naming the tensor, any buffer whose sha256 differs from the pin; no
    process rebuilds the record after the first launch.
  - A tied table (one Parameter read by a prologue and an epilogue module) whose two units fall on different stages
    is owned by the Head stage (the table, its optimizer state and its accumulator) and keeps the oracle's order: for
    microbatch i, `T_i` = the sorted unique ids of the microbatch (from `make_batch` on both ends; no index travels),
    `h_i` the Head use's gradient (dense), `e_i` the embedding use's. When `h_i` exists the Head folds every row
    outside `P ∪ T_i` as `acc + (h_i + 0.0)` (the assignment `h_i + 0.0` at i = 0), where `P` is the union of rows held
    for microbatches whose `e` has not arrived, and holds `h_i` on the rest; when `e_j`'s rows arrive it resolves the
    oldest held microbatches in index order, `acc + (h_j + e_j)` on their held rows (`e_j` is +0 outside `T_j`).
  - Stage 0 holds a working copy of the table: the rows of the step's ids, NaN elsewhere. It runs HF's own embedding
    module and that module's own backward, refuses (`TIED-SPARSE`, fatal) an `e_i` with a nonzero or negative-zero
    element outside `T_i`, and when its `B_i` ends sends `TIED(t, i)` holding `e_i[T_i]` in the forward direction.
    After the table's AdamW the Head sends `TIED(t+1, k)` for k = 0, 1, … in the backward direction, holding
    `R_k = T_k \ ∪_{j<k} T_j` in ascending id; stage 0's `F_k` of step t+1 waits for `R_0..R_k`, and a resume sends
    them from the loaded table before its first step. `TIED` frames are not lanes of `FWD`/`BWD` frames (`e_i` exists
    only after `FWD_i` left; rows inside step-(t+1) `BWD` frames would wait for the F they feed). Every stage between
    forwards each `TIED` fragment as it arrives, never waiting for its own compute, under the chain's (step, mb) rules
    and sockets (RUNTIME-I16 unchanged); a missing, duplicated, misshapen or out-of-order `TIED` frame is fatal, and so
    is any `TIED` frame received under a plan with no tied relay (validated at receipt, before its payload is used). This
    is specified for width 1 at stage 0 and at the Head stage only; where either has width > 1 the routing of `TIED`
    rows to the table's fold owners and back is not specified (ADR-0009 R14, open), and such a plan is refused,
    naming it.
  - A plan whose prologue and epilogue units share a stage sends no `TIED` frame. A frozen tied table (LoRA, which
    ADR-0009 refuses under `arch="hf"` until its targets are defined) has no accumulator: stage 0 keeps a read-only
    copy and no `TIED` frame travels.
- **RUNTIME-I20.** `optimizer` "adamw-host" (`runtime/optim.py`) keeps AdamW's moments in host memory and is
  byte-identical to the oracle built with the same optimizer. The device holds the fp32 parameters and the folded
  gradient (8 B per trainable element instead of 16); the moments stream through the device in chunks. The update uses
  single-rounding operations only (mul, add, sub, tensor-by-tensor div, sqrt; no lerp/addcmul/addcdiv, no division by a
  scalar), so its bits are independent of the chunking, the thread count and the vector width. It is a recipe pin
  (`pins()["optimizer"]`, present only when not "adamw"), named in the reference digest and in the checkpoint's param
  group (`"moments": "host"`); a state of another recipe is refused on load. Not combined with shard "pieces".
- **RUNTIME-I21.** A width-1 slot folds W's gradients as they complete. When the microbatch is next in fold order (and
  the slot is not the Head of a split tied table), `wgrad.flush_each` (flush's GEMMs, widenings and sums in store
  order, each entry's tensors released once run) yields each parameter's gradient at its last contribution and
  `OrderedAccumulator.fold_part` folds it at once (clone at the first microbatch, `add_` after); B's gradients follow.
  The running sum is bit-identical to the whole-set `add`; the stage's full gradient set is never resident beside it.
  An out-of-order W keeps the whole-set path.

- **RUNTIME-I22.** A stage builds only its own entries. `build_layers(cfg, seed, span)` for the builtin and general
  archs constructs every entry in order under the seed and drops each one outside `span` at once (`nn.Identity` in its
  place); under `init_scheme` "gpt2" a dropped entry's normal draws fill a scratch tensor of the same shape. The kept
  tensors and the generator's final state equal the full build's bit for bit; one foreign layer is resident at a time.
  The stage, the fit tool and the profiler build their span; a tied table, LoRA and arch "user" build in full.

- **RUNTIME-I23.** Under "adamw-host" a width-1 builtin stage (no tied relay, no MoE, not pieces) runs the step's AdamW
  beside the next step: a side thread (side CUDA stream) updates the parameters in chunk order and releases chunk
  layer li once all its parameters hold the update; the next step's forward waits per layer. Checkpoints, the run's
  end and failures join it first. The ops and their order are the stepping-first ones, so the bits are too.
- **RUNTIME-I24.** "adamw-host" moves the moments through three CUDA streams per step: a chunk's copy in, its update
  (on the step's stream) and its copy out are ordered by events through `slots` device staging buffers, so the next
  chunk's copy in and the last one's copy out overlap the update, and a layer's release (an event the forward's
  stream waits on) never waits for a copy out. Before a process's third optimizer step a CUDA stage moves the
  moments of parameters, in order, onto the device while they fit 85% of the memory free before its build less the
  reserved peak so far and 1 GiB (`WT_OPT resident`; WANTRAIN_OPT_RESIDENT=off disables it); a checkpoint carries
  host copies. The update's ops and values are the same wherever the moments live and however their copies are
  scheduled, so the bits are too. Measured 2026-10-12 (13.5B, two H100 NVL): the host round trip, 16 bytes per
  parameter per step, held each step's first forward for 2.7 s of a 12.7 s step.
- **RUNTIME-I25.** `recompute_stages` (per stage; None follows `recompute`) lets a builtin stage keep F's graph instead
  of rebuilding it in B. The coordinator sets it from the solved orders (`zero_bubble.selective_recompute`): a stage
  whose order holds at most one microbatch between an F and its B, always the last, keeps its graph, since B's rebuild
  would hold that same graph for that one microbatch. A stage a measured fit finds over budget recomputes again before
  any budget is calibrated; WANTRAIN_SELECTIVE_RECOMPUTE=off disables it. A builtin forward gives the same bits with
  or without its graph (the `recompute_last_keeps_graph` and `recompute_mixed` bitwise arms), so it is no pin; an arch
  "hf" run takes one value for every stage.
- **RUNTIME-I26.** A width-1 builtin dense chain (no wide stage, pieces, hf, tied table or MoE) can be re-cut at a
  checkpoint wave: `runtime/recut.recut_states` moves every entry's parameters, buffers and optimizer state (mapped
  by entry and parameter name, the stage's order) to the stage that now holds it, the same tensors renumbered, the
  pins naming the new partition, the losses merged. The coordinator writes the converted waves in place of the
  common wave (`CheckpointStore.replace_wave`), drops the newer ones, and relaunches under a new tag
  (`<tag>.cut<k>`): a worker never holds two different waves of one step under one tag. Resumed under the new cut,
  the run continues the uninterrupted run's losses exactly (`test_recut_resume_continues_the_sequence_bit_for_bit`,
  adamw and adamw-host).
- **RUNTIME-I27.** A deferred W that fills a network wait runs in sub-blocks (`_WTask.run(deadline)`, slices of
  `W_SLICE_S`, 5 ms): between gradients it reads the inbox and stops as soon as the op the order waits on can start,
  resuming where it stopped at the next wait or at its own turn. Each parameter's gradient is still one accumulation of
  the same terms in the same order, so the bits are the unsliced W's (`test_a_w_split_into_sub_blocks_while_filling_waits_is_bitwise`). The task
  holds only the tensors it needs and drops them when it finishes (a per-call closure kept the step's accumulator
  alive: +1.45 GiB per step, 2026-10-13). One optimizer side stream per run (a new stream per step grew the caching
  allocator's reserve, its blocks cached per stream).
- **RUNTIME-I28.** A streamed W hands each single-contribution weight gradient to the fold in its GEMM's output dtype
  (`wgrad.flush_each(widen=False)`); the accumulator keeps the running sum in the parameter dtype, widening the first
  part by an exact copy and every later one inside `add_`, which widens its operand exactly before the same add. The
  sums are the widen-first fold's bit for bit (`test_a_narrow_gradient_folds_to_the_widened_sum_bit_for_bit`); the
  widened copy (6 bytes per parameter of memory traffic per microbatch, 2.3 GB per d=5632 layer) is never made, and a
  part held for its predecessor stays narrow.

## 5. Ordered task list (future milestones)

1. **M2 — runtime telemetry for re-planning.** Emit measured per-op F, B and W times and per-hop transfer times in
   one `WT_TELEM` line per step (RUNTIME-I9) (median per step), so the coordinator can re-solve from runtime costs instead of the profiler. The profiler
   underestimates by about 30% on CPU.
2. **M2 — between-step ownership change.** Accept a new `mb_owner` and `schedule_order` at a step boundary (pushed by the
   coordinator through a restart-in-place with resume). The fold order is unchanged, so RUNTIME-I1 holds.
3. **M3 — ZB-V execution (RUNTIME-I8).**
   - Each rank hosts two chunks `[a0, b0)` and `[a1, b1)`.
   - Ops carry a chunk index: `[op, mb, chunk]`.
   - Routing for chunk 1 goes in reverse rank order.
   - The checkpoint holds both chunks.
   - Requires SPEC-scheduler task 3.
4. **M4 — slice-only build (RUNTIME-I14, ADR-0007; with SPEC-model task 1).** `StageRuntime.__init__` builds through
   `build_slice` at step 0, or through the re-tied skeleton and the wave on resume (`arch="user"`: the full build and the
   wave); `_resume` refuses a wave without `rng`.
   `JobSpec` gains `build: "slice" | "full"` (default `"full"` until ADR-0007's E1–E5 pass, then `"slice"`) and
   `init_record`. `validation/state.distributed_manifest` takes names, shapes and sharing from `skeleton(cfg)` (under `arch="user"`,
   which has no skeleton, from today's full build) and
   records the init record's sha256, binding and fingerprint; `validation/reference.py` builds with `build_layers` (the
   phase-major walk of `init_phases`) and checks its ledger states and digests against the record when there is one.
5. **M2b — exact sharded stages (ADR-0006; RUNTIME-I10–I13).** Two `JobSpec` fields, each rolled back alone.
   - Step 1, `wire: "native"` (also valid with `shard: "zero1"`): W iterates `wgrad.flush_iter(store, order)`
     (MODEL-I7, `order` from the plan) and ships each yielded gradient's non-owned part at once, in the dtype it was
     produced in, through per-destination send queues served need-first (RUNTIME-I13). The compute stream records an
     event per yielded gradient; a transport thread copies it D2H on a copy
     stream into pinned buffers, in chunks of at most 16 MiB, and sends each chunk when its copy completes. Received
     chunks go H2D on the copy stream; the fold waits for the copy on the device, never on the host. The owner widens
     to fp32 before `OrderedAccumulator`. The step barrier stays.
   - Step 2, `shard: "pieces"`: every trainable tensor of a width-w stage is cut along dim 0 into w pieces whose
     boundaries are multiples of `Q_t = lcm(row length, 128)` elements (equal by default; the layout records explicit
     boundaries). Peer k owns piece k of every tensor: fp32 master, AdamW m and v, fp32 fold accumulator.
   - Every peer holds a step-resident working copy of every tensor in its consumer dtype: the recipe's compute dtype
     (`precision.spec(compute)`) for `wgrad.owned_params` when the recipe autocasts, the master dtype otherwise.
   - Exchange units are `SplitLinear` modules (and each block's small tensors); in an MoE block the E experts'
     `SplitLinear`s of one projection (gate, up, down) form one unit, each tensor cut into its own pieces and framed
     in expert index order; the layer's selection bias belongs to its small-tensor unit and its int64 count vector
     rides that unit's GRADS frame in a lane of its own (RUNTIME-I18) (ADR-0008, M4; a change to M2b's built
     `exchange_units` and `check_frame` tables for MoE layers only, a dense model's tables unchanged). Under HF modules
     (ADR-0009) each 3-D expert tensor is one unit: `gate_up_proj` and `down_proj`, two routed-expert units per MoE
     layer, cut along dim 0 at `Q_t = lcm(trailing product, 128)` (whole experts); their gradients are produced in B
     in the master dtype (fp32 under today's recipe) and ship when W(mb) starts, as the Embed's tables. On both paths a
     shared expert's `SplitLinear`s are one unit each, and the router weight joins the small-tensor unit. The Embed's tables (`tok`, `pos`) are
     one unit of their own, first in forward unit order; their fp32 gradients, produced in B, ship when W(mb) starts.
     The Head's norm rides with the Head's linear. Per unit and destination, one frame per microbatch. A unit's update runs in the first `_pump` after its fold completes (RUNTIME-I13); the owner casts its
     piece once and sends it; gathers go in forward unit order; step t+1's F on a unit waits for that unit's copy.
     `_optimizer_step_wide`'s PARAMS barrier is removed except at checkpoint, re-plan and last steps.
   - W never waits on the network: no credits. Out-of-order remote arrivals wait in host memory and move to the device
     only to be folded; the peer's own pieces stay on the device until their prefix is folded. Both holds are bounded by
     one step, per peer (RUNTIME-I13), which the planner charges in full (SCHEDULER-I16); there is no runtime overflow
     path. `shard: "pieces"` requires `wire: "native"`; JobSpec refuses `{wire: "fp32", shard: "pieces"}`.
   - Re-planned ownership moves piece state (master, m, v) at a barrier step, as a byte permutation.
   - Under `shard: "pieces"` the identity manifest hashes the logical state in canonical name order and records each
     peer's final working-copy digest per tensor; distinct digests are a split that never compares equal. Under
     `shard: "zero1"` it keeps `names[p::w]` and the replica-agreement check.

6. **M4 — exact MoE (RUNTIME-I18, ADR-0008).** The aux record and `MOE-REPLAY` check in `stage.py` (F records,
   B's rebuild compares); the selection-bias update after the step in `stage.py` and `validation/reference.py`; the
   count transport of RUNTIME-I18 (zero1's ownership list with the biases appended, the count vectors in the shard
   gradient messages to the owner and the bias in its broadcast, the frame checks); the `aux` and `moe_bias` sections
   in `validation/state.py`
   (`reference_manifest` and `distributed_manifest`, gathered from every slot's checkpoint); the expert units of
   task 5 on M2b's merged code.

7. **M4 — any HF decoder (RUNTIME-I19, ADR-0009; with SPEC-model task 4).** Stages of `arch="hf"` build their units
   through `hfm` and assert the determinism flags at start; the row-sparse tied fold in `OrderedAccumulator` and the
   `TIED` frames, forwarded cut-through by intermediate stages; the buffer record in the JobSpec, the state file and
   every wave, checked at every load; the HF oracle in `validation/reference.py` (units placed by the device map,
   buffers from the record); each piece reachable only through `arch="hf"` (a tied relay only in an hf plan), so a
   rollback reverts each piece by id to its own target (ADR-0009 `round6.md` §4): 7a (row-sparse fold) and 7b
   (`TIED` relay) to "tied HF models at S > 1 refused at planning", any `TIED` frame fatal; 7c (buffer record in
   JobSpec, state file, waves; a wave carrying one is then refused on resume) and 7d (HF oracle) to SPEC-model 4a;
   `setup_determinism` assigns its pins (a pre-set differing value refused, naming it) and reads every determinism
   flag back, `allow_tf32` included (ADR-0009 house item, its own test: `covers: RUNTIME-I19`, `WOULD-FAIL-IF:` a
   pre-set `ATEN_CPU_CAPABILITY` survives, or `allow_tf32` is True after setup);
   tiny HF families (llama, gemma2 tied, granite, qwen3) in the `oracle` and `identity` rungs; ADR-0009's E3 and E4
   first.

## 6. Decisions

- **One execution model.** 1F1B, the greedy event loop, interleaving and the two-call backward were removed. Rejected:
  keeping them as switches; every switch doubled the test matrix and the slowest path set the bar.
- **Static IWRR ownership over a dynamic claim pool.** A pool cannot be scheduled ahead. Rejected: work stealing, which
  is non-deterministic in timing and makes the solved order meaningless.
- **Process-wide dW store.** CUDA and MPS run backward on autograd device threads. Rejected: a thread-local store, which
  silently dropped dW.
- **A missing shard on resume is fatal.** Rejected: starting the rank at step 0, which deadlocks the pipeline.
- **A tied table across stages keeps the stock order by a row-sparse pending fold on the Head stage, its rows
  relayed on the chain in `TIED` frames (ADR-0009).** The embedding use is +0 outside the microbatch's ids, so the Head folds those rows
  as soon as its own use exists and holds only the rest (≤ `n_pend·min(V, n_pend·T_mb)·h·4` B, against one dense
  table per microbatch in flight); equal to the stock fold in every measured case (ADR-0009 Evidence 12). Rejected:
  the per-use fold `(Σe)+(Σh)` (a recipe change that differs from every stock run); dense pending tables (45 GB at
  Gemma3-27B); shipping the summed table each step (≥ 3.8 s at 10 Gbps for Gemma2-2B); the Head computing the
  embedding output per microbatch (both directions per microbatch); a direct Head↔stage-0 link (DESIGN §10's chain,
  a topology decision of its own); lanes inside `FWD`/`BWD` frames (`e_i` exists only after `FWD_i` left, and the
  next step's rows would wait for the F they feed).
