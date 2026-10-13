# Current architecture: what exists

This describes the distributed-training engine `wantrain` (`src/wantrain`) as it exists on `main`, as merged in `9f7d81c`.
It gives every component's responsibility, dependencies, interfaces, data and control flow, and failure, security and
resource boundaries. Workflows read it to *understand before modifying*: what is described here is preserved unless
evidence shows it cannot meet a requirement.

Other references: the file-level map is [ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md); the narrative and running guide is
[docs/engine/ARCHITECTURE.md](../engine/ARCHITECTURE.md); the provenance (what came from Elysium, what was fixed) is
[docs/engine/MIGRATION.md](../engine/MIGRATION.md).

## 1. System in one paragraph

The engine performs synchronous, lossless pipeline-parallel training of a decoder across machines joined by wide-area
links. The model is a flat list of layers `[Embed, Block × n, Head]`, cut into contiguous slices, one slice per
**stage**. A stage can be **wide**: several **peers** hold the same slice and split the microbatches. One process runs
per **slot** `(rank, peer)`. Every slot executes a precomputed **zero-bubble** order of three operations per owned
microbatch:
- **F**: forward;
- **B**: input gradient only, on the critical path;
- **W**: deferred weight gradient.

The engine's contract is that the loss at every microbatch is **bit-identical** to a single-device run of the same
recipe (the oracle). This holds for every partition, schedule, network condition, crash and heal. One **coordinator**
plans and supervises but never touches numerics; one **agent** per machine supervises its slot's process. Training
traffic flows slot to slot over TCP.

## 2. Processes and wires

| Process | Started by | Talks to | Wire |
|---|---|---|---|
| Coordinator (`wantrain coordinator`) | operator | every agent | control channel: JSON frames over TCP, HMAC challenge with a shared token (`transport/control.py`) |
| Agent (`wantrain worker`) | operator, one per machine | coordinator; its own stage process (pipes, files) | control channel; local subprocess |
| Stage (`python -m wantrain.worker.stage_main`) | agent (or `launch/local.py`) | adjacent-stage slots; intra-stage peers | data plane: framed tensors over TCP (`transport/framing.py`), peer-authenticated (`transport/identity.py`) |
| Probe echo servers | agent | other agents | raw TCP for RTT and bulk throughput (`resources/links.py`) |

**Data-plane topology.**
- **Width-1 chain** (`_connect_chain`): rank `r` dials `r+1` and accepts `r-1`, with no wrap link. Every hop is one duplex
  authenticated connection, dialed in a single direction (this works through asymmetric NAT). With `p2p_streams > 1`,
  fragments are spread round-robin over parallel flows per direction, with exact reassembly.
- **Wide stages** (`_connect_mesh`): every peer of stage `r` connects to every peer of stage `r+1`, plus an intra-stage
  clique for the ZeRO-1 exchange.
- **Launch order:** last rank first, so every listener is up before it is dialed.

**Frames** (`framing.py`): `FWD`, `BWD`, `END`, `GRADS`, `PARAMS`, `LOSS`, `HELLO`. The header carries step, microbatch,
chunk (the sending peer), shape and the fragment index and count. Codecs are `RAW_F32`, `RAW_I64`, `FP8_PLANES`, `OBJ`,
`F32_PLANES` and `PLANES`. `OBJ` uses the exact, code-free `serial` codec, never pickle. Payloads are capped by
`MAX_PAYLOAD`. `END` plus a half-close is the graceful-shutdown handshake: a reader drains until every neighbour's `END`.

**Control messages** (agent ↔ coordinator):
- agent → coordinator: `HELLO`, `HEARTBEAT`, `PROGRESS`, `TELEMETRY`, `EXIT`, `CKPT`, `RESULT`, plus a `*_RESULT` or
  `*_ACK` reply for each request;
- coordinator → agent: `CHALLENGE`, `WELCOME`/`REJECT`, `ASSIGN`, `PROFILE`, `FIT`, `MEASURE`, `PROBE`, `ABORT`,
  `RESTORE`, `RELEASE`, `SHUTDOWN`.

**Stage markers** on stdout, parsed by the agent:
- `WT_STAGE`, `WT_PHASE`, `WT_STEP <n>` (per step, on every rank);
- `WT_METRIC`, `WT_MEM`, `WT_WARN`, `WT_RESUME`, `WT_DONE <rc>`;
- `WT_PROFILE {...}`, `WT_FIT {...}`.

## 3. Components

### 3.1 Runtime: `runtime/`, `worker/stage_main.py`, `launch/`, `validation/`

- **Responsibility.**
  - Execute one slot's `schedule_order[rank][peer]` for every step, then fold gradients in microbatch order and take the
    optimizer step.
  - The local optimizer is AdamW. In wide stages it is ZeRO-1: peer `k` owns `params[k::w]`; shards are summed in index
    order and broadcast.
  - Checkpoint every `ckpt_every` steps: shard, optimizer and loss history.
- **Interfaces.**
  - `JobSpec` (JSON) in.
  - Data-plane frames in and out.
  - `WT_*` markers out, plus a results file with hex-exact losses on the last stage.
  - `StageRuntime(spec, rank, peer).run()`.
- **Flow per step.** `_run_step` walks the order:
  - **F** needs the activation from upstream; stage 0 needs no input.
  - **B** needs the gradient from downstream. The last stage starts from the loss. B computes `dX` and captures each split
    linear's `(X, dY)` (`model/wgrad.py`). `dX` is sent upstream at once and the graph is freed.
  - **W** flushes the captured `dW = dYᵀX` and stores the result for the ordered fold.
  - While blocked on a late input, an already-runnable W is pulled forward (`_fill_w`, when `schedule_fill`).
  - Leftover deferred W at the step end is an error.
- **Depends on.** model (layers, wgrad, precision), transport (frames, identity, shim), codec, scheduler (only the
  default order, through `JobSpec.finalize`).
- **Failure boundary.**
  - A neighbour EOF without `END`, a dial or auth failure, or a schedule-order violation is fatal to the slot (non-zero
    exit). The control plane recovers from it.
  - `resume=true` with a missing shard is fatal on purpose: a rank at step 0 next to peers at step N would deadlock.
- **Security boundary.** Peers are authenticated (ed25519 allowlist, or HMAC with a token). Payloads are bounded.
  Nothing is unpickled. Traffic is **not encrypted**.
- **Resource boundary.** One process per slot, with one device (`devices[rank][peer]` or `device`).
  - **Memory:** activations of in-flight F (the boundary only, when `recompute`) plus pending W inputs (`16·h` values per
    token per block).
  - **Network:** one boundary tensor per microbatch per direction per hop.
- **Oracle.** `validation/reference.py` trains the same recipe on one device with the same microbatch order.
  `validation/compare.py` diffs the loss sequences in hex. `launch/local.py` runs every slot locally and diffs.

### 3.2 Model: `model/`

- **Responsibility.** Build deterministic flat layer lists (built-in GPT, Llama-family pieces, HF checkpoints, or a user
  `build(cfg)` with a validated contract). Also: precision recipes, fp8 boundary scaling, LoRA, deterministic data and
  corpora, and the deferred-dW machinery.
- **Interfaces.**
  - `build_layers(cfg, seed)`, `loss_fn`.
  - `wgrad.capture()` and `wgrad.flush(store)`; `wgrad.convert(module)`, which skips tied weights.
  - `precision.spec/classify`, `make_batch(data_seed, step, mb, ...)`.
- **Invariants it carries.**
  - The same seed builds the same weights on every host.
  - `SplitLinear` gives bit-identical gradients to `nn.Linear`.
  - The batch for `(step, mb)` depends on nothing else.
- **Resource boundary.** Today every stage builds the full model and then slices it, so host RAM caps the model size.

### 3.3 Scheduler: `scheduler/`

- **Responsibility.** Decide where and how a job runs:
  - node selection (`selection.py`, with `linkmemory.py`);
  - chain order by worst hop (`partition.order_stages`, `best_chain`);
  - the layer cut: exact min-max DP over per-layer compute costs, with a memory predicate (`partition.assign_layers`);
  - placement and peer weights (`placement.place`);
  - the precision ladder against the link (`autoconfig`);
  - microbatches per step from the solved schedule's bubble (`schedule.solve_microbatches`);
  - plan mode (`planner.plan_job`);
  - the zero-bubble solve (`zero_bubble.solve` and `schedule_for_spec`).
- **Zero-bubble solver.** A port of the reference automatic heuristic: warm-up F bounded by memory until the first B can
  arrive, then alternating F/B, W inserted where it fits, a grid over three choices, and selection by the simulator.
  It is generalized to per-stage F/B/W costs and per-hop communication. Warm-up counts are clamped to the upstream
  stage's. For wide stages, stage-level orders are split per peer by IWRR ownership; each peer keeps the stage order's
  relative sequence, so the result cannot deadlock. `simulate_slots` checks the split.
- **Interfaces.**
  - Pure functions over `JobSpec`, `ZBConfig`, `NodeInfo` and measured links.
  - `schedule_for_spec(spec, profiles[stage][peer], hops, budgets, mem_mode)` fills `spec.peer_weights`, `mb_owner` and
    `schedule_order`.
- **Failure boundary.** Plans are refused with the number that refused them (link floor, memory). A partition the
  memory predicate cannot satisfy falls back with a warning. The solver raises when no schedule fits the memory budget.
- **Resource boundary.** CPU-only, in the coordinator. The solve is milliseconds to seconds for M ≤ 1024.

### 3.4 Transport and codecs: `transport/`, `codec/`

- **Responsibility.** Frames and fragments, the exact object codec, peer identity, the control channel, WAN emulation,
  and lossless boundary codecs (byte planes + zstd; scaled fp8 for fp8 recipes).
- **Invariants it carries.**
  - Every codec round trip is byte-exact.
  - Fragments reassemble to the exact payload in any arrival order.
  - `serial.loads` never executes code.
  - A handshake binds identities, and the allowlist refuses unknown peers.
- **Security boundary.** This is the only code that parses bytes from the network: payload caps, typed decoding,
  authentication before data.

### 3.5 Control plane: `coordinator/`, `worker/agent.py`, `fault_tolerance/`, `resources/`

- **Coordinator lifecycle.**
  - **Discovery:** agents register with their probe results.
  - **Selection:** spares are the rest of the pool.
  - **Routing:** probe matrix → chain → bidirectional hop measurement → re-route on disagreement.
  - **Planning:** recipe or plan mode, then an on-device fit check.
  - **Profiling:** every slot times F, B and W on its own device (`PROFILE`).
  - **Cut refinement:** applied if it gains more than 5%.
  - **Zero-bubble solve:** from profiled costs, hop comm and budgets; the rationale is logged.
  - **Launch:** last rank first.
  - **Monitor**, then **heal** or **restart**, then **complete** (hex losses and telemetry).
- **Liveness judge** (`liveness.py`) verdicts:
  - **ok**, **done**;
  - **suspect**: inside the reconnect grace nothing is judged;
  - **convict**: control connection lost past the grace; stage gone without its done marker; a peer-lost report that
    points at the silent slot; two heartbeat strikes;
  - **stalled**: everyone alive, no progress → restart in place;
  - **failed**: an engine exception, or simultaneous losses, which are treated as a network problem and never as several
    dead boxes.
- **Heal.**
  1. Abort the survivors; each ships its last wave, sha256-verified.
  2. Pick a spare (`find_replacement`: identical model first, then the region).
  3. Take the newest wave that every slot holds, and restore it.
  4. Relaunch with `resume=true` under the original spec. Partition, schedule, ownership and seeds stay pinned.

  Budgets: `max_heals`, `max_restarts`.
- **Agent.** Supervises the stage subprocess. It reports `alive` until the exit is reported (no liveness race),
  reconnects with backoff, and ships waves as files over the channel.
- **Security boundary.** Coordinator admission is token-HMAC. Without a token the coordinator warns and accepts any host.
  The coordinator is a single process (a single point of failure). `AuditPolicy` (verify-before-fold against malicious
  workers) exists as a library but is not wired.
- **Resource boundary.** The coordinator holds every checkpoint wave on its disk (the newest common waves, pruned).

## 4. Configuration of record

A run is fully determined by one `JobSpec` JSON.
- **Recipe:** `model`, `init_seed`, `data_seed`, `lr`, `steps`, `microbatches`, `mb_size`, `recompute`, `dataset_path`,
  `data_shards`.
- **Topology:** `stages`, `stage_widths`, `partition`, `peer_weights`, `mb_owner`, `devices`.
- **Schedule:** `schedule_order[stage][peer] = [[op, mb], ...]` and `schedule_fill`.
- **Transport:** `endpoints`, `trusted_peers`, `p2p_streams`, `wire_codec`, `adapt_budget_ms`, `frag_min_bytes`,
  `dial_timeout_s`, `links` (emulation).
- **Checkpoint:** `ckpt_every`, `resume`.

`finalize()` derives every missing field deterministically (analytic schedule, IWRR ownership) and validates coverage.
Each op must cover exactly the slot's microbatches.

## 5. Known limits of what exists

These are tracked as gaps in `DESIGN.md` §8:
- no ZB-V schedule;
- no re-partitioning or rebalancing during a run;
- the solver refuses rather than degrades under a tight memory budget at high latency;
- a wide-stage peer loss needs a checkpoint resume (the shard is not replicated);
- plan mode supports width-1 only;
- `AuditPolicy` is not wired;
- the data plane is not encrypted;
- the coordinator is a single point of failure;
- every stage builds the full model;
- no validation yet on real GPUs across regions.

The tests themselves are wall-clock timed: a sleeping host shows up as 900 s stalls, not as a failure of the engine.
