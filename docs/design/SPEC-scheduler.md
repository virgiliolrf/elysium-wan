# SPEC-scheduler: where, how much and in what order

**Workstream:** `scheduler`. **Owned paths:**
- `src/wantrain/scheduler/`;
- `tests/unit/test_zero_bubble.py`, `tests/unit/test_schedule.py`, `tests/unit/test_scheduler.py`.

## 1. Scope

The scheduler holds every decision the coordinator takes before launch, as pure functions:
- node selection and link memory;
- chain order;
- the layer cut;
- placement and peer weights;
- the boundary precision;
- microbatches per step;
- the zero-bubble order for every slot.

It never runs training and never touches numerics. Its outputs only change *when* things run.

## 2. Domain model (normative)

### 2.1 Zero-bubble solver (`zero_bubble.py`)

A port of the automatic heuristic of Qi et al. ("Zero Bubble Pipeline Parallelism", ICLR 2024; sail-sg `zb.py`).

- **Inputs** (`ZBConfig`):
  - `n_stages` `p` and `n_micro` `M`;
  - per-stage costs `cost_f`, `cost_b`, `cost_w` (seconds);
  - per-hop `comm[j]` for hop `j → j+1`;
  - per-op memory `mem_f`, `mem_b`, `mem_w` per stage, which sum to 0 per microbatch;
  - budgets `max_mem[stage]`.
- **Heuristic** (`_manual_order`):
  1. Warm-up: stage `j` runs as many F as memory allows until its first B can arrive. Its first B arrives after
     `Σ_{k≥j} (F_k + comm)` forward and `Σ (B_k + comm)` back.
  2. Warm-up counts are clamped so that `f_required[j] ≤ f_required[j-1]`. A stage never runs more F than its upstream
     produced. This clamp is the heterogeneous fix; uniform costs never trigger it.
  3. Steady state: alternate F and B.
  4. Insert W where it fits before the next input arrives.
  5. Keep each stage at least one F ahead of the next.
  6. Drain the remaining W at the end.
- **Search.** A grid over `allow_bubble_before_first_b`, `prioritize_b` and `no_bubble_greedy`. The winner is the
  smallest makespan under `simulate`.
- **Warm-up family (SCHEDULER-I13).** `L_j` is the largest warm-up whose memory replay of `_warmup_order(M, k)` (k F,
  then B, W and the next F per microbatch) fits stage `j`, capped by `L_{j-1}`. For gaps `g` in a geometric range
  `1 … g_max = ⌈(L_0 − 1)/(p − 1)⌉`, the candidate is `k_j = min(L_j, 1 + g·(p − 1 − j))`: non-increasing, at least 1,
  `g = 1` is 1F1B's warm-up. Why gaps and not the largest warm-ups: equal warm-ups make every microbatch wait a hop's
  round trip (stage `j+1`'s next F follows stage `j`'s B), so the gap between neighbours is what hides `2·comm`; the
  largest warm-ups fit and idle (7B example at 8 GB: `[316]×8` replays to a 98.28% bubble, the gap search to 17.31%
  with `[85, 73, …, 13, 1]`). `simulate` picks the shortest makespan among the gaps.
- **Selection.** The returned schedule is the shortest simulated makespan among the grid candidates whose memory replay
  fits and the warm-up family's best (ties keep the grid's). It is the best of these two candidate sets at the given
  budget, not a global optimum.
- **Memory modes.**
  - `1p`: the budget equals 1F1B's peak, `p · per-microbatch`.
  - `2p`: double that budget.
  - `given`: explicit `max_mem`.
  - `auto`: use the measured budgets when present, else `2p`.
- **Simulators.**
  - `simulate(cfg, orders)` executes stage-level orders. It returns the makespan, or raises on deadlock or incomplete
    coverage.
  - `simulate_slots(cfg, slot_orders, mb_owner, peer_weights)` does the same per slot. A peer's op time is
    `stage_cost · Σw / w_peer`.
- **Wide stages.**
  - The stage-level order is split per peer by owner (`slot_orders`). Each peer keeps the stage order's relative
    sequence, so the split cannot deadlock.
  - Ownership is `iwrr(M, weights)`: interleaved weighted round robin, proportional to the weights, plain round robin
    when they are equal. This is SWARM's routing rule, made static.
- **Engine bridge.**
  - `schedule_for_spec(spec, profiles[stage][peer], hops, mem_budgets, mem_mode)`:
    1. Sets `peer_weights` from the profiled throughput `1/(t_F + t_B + t_W)`, normalized to the fastest peer.
    2. Sets `mb_owner` by IWRR.
    3. Builds each stage cost per op as the parallel-service cost `1 / Σ_peers (1/c_peer)`.
    4. Solves.
    5. Writes `schedule_order[stage][peer]`.

    It returns `{schedule, config, slot_makespan, peer_weights}`.
  - `default_schedule(spec)` solves from `analytic_costs(spec)` (FLOP-proportional F:B:W), `2p` memory and zero comm. It
    is deterministic.
- **Memory model** (`stage_memory_model`):
  - With recompute: `mem_f` = boundary bytes, `mem_b` = pending-W bytes (`layers·mb·seq·16·h·4`),
    `mem_w = -(mem_f + mem_b)`.
  - Without recompute: `mem_f` = the live graph, and B frees the graph down to the pending-W share.
  - (M4, ADR-0008) An MoE block's pending W per token is its attention's values plus `k·(2h + 3f)` for the routed
    experts plus `2h + 3f_sh` for a shared expert (SPEC-model §2), never a routing-dependent figure. Under HF modules
    (ADR-0009) `mem_b` adds each MoE block's held bank gradient, `b_m·E·3hf` bytes (SPEC-model §2), in both branches
    (with recompute beside the pending W, without beside the share B keeps), and `mem_w` releases it (`mem_w =
    −(mem_f + mem_b)` unchanged). The memory replay therefore counts it once per microbatch between its B and its W:
    `m_BW·b_m·Σ E·3hf` at a point where `m_BW` microbatches are in flight (a 4-layer Mixtral stage: 22.5 GB each).
- **Communication.** `comm_seconds(bytes, mbps, rtt_ms) = rtt_ms/2000 + bytes·8/(mbps·1e6)` seconds: the one-way
  latency plus the transfer at the link rate. Bytes are the boundary tensor in the recipe's boundary dtype.

### 2.2 Microbatches per step (`schedule.solve_microbatches`)

The smallest M whose solved schedule (ZB-2p) idles at most `bubble_target` of the pipeline. Per-microbatch costs are in
the ratio `fbw`, and each hop costs `comm_ratio · t_F`. The search walks M geometrically, then bisects. The token cap
(global batch) wins over the bubble target. `floor` is respected. One stage needs no pipeline.

### 2.3 Cut, order, selection, placement and precision

- **Layer cut** (`partition.assign_layers`):
  - The exact min-max dynamic program over contiguous ranges of `n_layer + 2` entries, minimizing `max_stage(cost/speed)`.
  - Per-layer costs come from `layer_costs`: embed `3h`, block `h(72h + 12s)`, head `6hV`. (M4, ADR-0008) An MoE
    block costs `6·P^F + 12·h·s`, with `P^F` its attention, router, k experts and shared expert: the parameters a
    token uses (the routed rows are `T·k` exactly, so this is not an estimate of the routing). It omits the per-expert
    launches and host count reads, so it is a lower bound of an MoE block's time; the solve uses profiled per-op
    costs, and the runtime re-plan corrects a cut it misbalanced. A 32k-vocabulary head costs
    about 3 blocks.
  - Optional floors, ceilings and a `fits(stage, a, b)` memory predicate. If no cut fits, the fallback ignores the
    predicate with a warning.
- **Chain order** (`order_stages`, `best_chain`): maximize the worst hop. Exact branch-and-bound up to 8 nodes, greedy
  best-next-hop from every start beyond that. The search routes around pairs with no connectivity.
- **Selection** (`select_nodes`):
  1. Filters: dead-host memory, device types, the capability window, memory and uplink floors. (The human,
     2026-10-11; CONTROL-I25) Every arch: a candidate fleet holds one device class (`NodeInfo.device_class`); if no
     single class can field it, selection refuses `MIXED-FLEET` with the per-class counts. `plan_job` and `place`
     refuse a fleet of more than one class (`MIXED-FLEET`, naming the classes) before any cost is computed.
  2. Group by region and hardware. With `SelectionPolicy.same_region` (the default), a fleet never spans regions; if no
     single region can field it, selection refuses with the per-region counts. Cross-region fleets exist only with
     `same_region=False`.
  3. Rank by measured path verdicts, then bf16 support, then total speed. Distinct addresses. The rest are spares.
- **Placement** (`place`): peer weights from node speeds, the cut by stage speed (the sum of its peers' speeds), links,
  then `spec.reschedule()`.
- **Precision admission** (`autoconfig`):
  - Every rung of `LADDER = fp32, bf16, fp16, fp8_e5m2, fp8_e4m3` is priced. The bandwidth floor is
    `bytes_per_token·mb·seq·8/window`, and the margin defaults to 2×.
  - Lossless rungs are preferred.
  - A refusal names the number that refused.
  - (M4, ADR-0009) For `arch="hf"` the boundary dtype is not chosen: it is HF's residual dtype under the recipe (fp32
    with fp32 parameters and autocast, the parameter dtype otherwise), and a declared `boundary_dtype` other than it is
    refused at validation, naming it; no rung of `LADDER` other than that dtype is priced.
- **HF models and tied tables** (M4, ADR-0009). An `arch="hf"` model is costed from its config through the same
  `[prologue, layer × n, epilogue]` entries (per-layer parameters from the skeleton's shapes; per-unit window costs as
  PROFILE measures them). Eager attention adds `b·H·S²·s` B per layer held for backward to the activation memory model,
  `s` = 6 under autocast with fp32 parameters, 4 in fp32, +2 (autocast) or +4 (fp32) with an attention softcap (all of
  a recomputed stage's layers at once). A tied table whose prologue and epilogue units fall on different stages charges the Head stage with the
  table, its optimizer state, its accumulator, one transient dense gradient and the held rows (`n_pend·min(V,
  n_pend·T_mb)·h·4` B, `n_pend` from the schedule), and stage 0 with the working copy and one transient dense
  gradient (`2·V·h·4` B); every chain hop between them with `|T_i|·h·4` B more per microbatch in the forward direction
  (`TIED` frames) and `|U_{t+1}|·h·4` B more per step in the backward direction; and the step with `ΔT_tied =
  M·max(0, (a+ē)/B_min − max(t_s, a/B_min)) + Σ_{k=1}^{S−1} max(0, |R_k|·h·4/B_min − t_F) + Σ_hops L_hop (both
  ways) + (|T_{M−1}|+|T_0|)·h·4/B_min + t_AdamW(table) + M·V·h·4/BW_mem + max(0, Σ_{k=1}^{M−1} |R_k|·h·4/B_min −
  W_idle)`, `W_idle = Σ_hops L_hop (both ways) + S·t_F + (S−1)·a/B_min + t_B` (`a` the cut's bytes per microbatch,
  `t_s`, `t_F`, `t_B` stage 0's per-microbatch times, the Head's cost including the tied head GEMM; `|T_i|`, `|R_k|`
  from `make_batch`'s ids; ADR-0009, `docs/research/adr-0009/round4.md` §1). `autoconfig.delta_t_tied(t_s, t_F, t_B,
  rows_T, rows_R, hops, latency_s, bandwidth_Bps, *, d_model, vocab_size, tokens_mb, mem_bw_Bps,
  adamw_bytes_per_elem=28)` computes it with one `L` per hop (`2·hops·L`), `a = tokens_mb·h·4`, `t_AdamW(table) =
  V·h·adamw_bytes_per_elem/BW_mem` (16 B read + 12 B written per element) and `BW_mem = mem_bw_Bps`, the Head
  device's memory bandwidth; `mem_bw_Bps = inf` (unmeasured) makes the scan and `t_AdamW` terms 0, a lower bound.

## 3. Interfaces

**Provides**
- `ZBConfig`, `Schedule`, `zero_bubble(cfg, mem_mode)`, `simulate`, `simulate_slots`, `bubble_rate`, `iwrr`,
  `slot_orders`, `solve`, `default_schedule(spec)`, `schedule_for_spec(spec, profiles, hops, mem_budgets, mem_mode)`,
  `comm_seconds`, `hop_comm_seconds`, `analytic_costs`, `stage_memory_model`.
- `solve_microbatches(n_stages, bubble_target, tokens_per_step_cap, mb_size, seq_len, floor, fbw, comm_ratio, cap)`,
  `solve_fleet_ceiling`, `solve_k_inflight`.
- `assign_layers`, `layer_costs`, `proportional_partition`, `order_stages`, `best_chain`.
- `select_nodes`, `candidate_fleets`, `SelectionPolicy`, `LinkMemory`.
- `place`, `stage_speeds`, `memory_order`, `plan_job`, `Fleet`, `Overrides`, `stage_fits`, `evaluate`, `choose`, `plan`,
  `build_spec`, `PRODUCTION`.

**Consumes:** runtime `JobSpec` (fields only), model `GPTConfig`, `precision`, control `NodeInfo`.

## 4. Invariants

- **SCHEDULER-I1.** On the reference's own inputs (uniform costs, zero comm), the solver reproduces the reference
  implementation's schedules exactly (golden orders).
- **SCHEDULER-I2.** The simulator reproduces the paper's bubble formulas for 1F1B, ZB-H1 and ZB-H2.
- **SCHEDULER-I3.** For any heterogeneous per-stage costs, per-hop comm and memory budget, the solver returns a complete,
  dependency-respecting schedule within the budget, or raises the documented memory error. It never returns an invalid
  order.
- **SCHEDULER-I4.** The simulator rejects any order that deadlocks or misses an operation.
- **SCHEDULER-I5.** Wide-stage slot orders, split from a valid stage order by any owner map, cannot deadlock.
- **SCHEDULER-I6.** IWRR ownership is proportional to the weights, interleaved, and plain round robin when the weights
  are equal.
- **SCHEDULER-I7.** `default_schedule(spec)` is deterministic and complete, and survives a JSON round trip.
- **SCHEDULER-I8.** `solve_microbatches` returns the smallest M whose solved schedule meets the bubble target. The
  token cap wins and the floor is respected. With 4 stages the solved schedule needs far fewer microbatches than the 1F1B
  fill/drain (7 vs 57 for ≤ 5%).
- **SCHEDULER-I9.** The layer cut is the exact min-max (equal to brute force), and it respects the memory predicate
  whenever any cut does.
- **SCHEDULER-I10.** The chain order maximizes the worst hop (equal to brute force on small meshes) and routes around
  dead pairs.
- **SCHEDULER-I11.** Selection prefers one region, measured paths and fast nodes. It keeps spares and distinct hosts,
  and refuses cross-region when no single region can field the fleet.
- **SCHEDULER-I12.** The planner refuses a link nothing fits, and a cut a small card cannot hold, naming the number.
- **SCHEDULER-I13.** Under an activation budget too tight to hide the latency, the solver degrades to the best schedule
  that fits (a larger bubble) instead of refusing, and reports the bubble. "Best" is the Selection rule of §2.1 (grid
  and warm-up family at the given budget). Guaranteed: every returned schedule's memory replay is `≤ max_mem` on every
  stage, `simulate` accepts it, the bubble is in `params`, and the only refusal is a budget below one microbatch, naming
  the stage, the bytes and the budget. Not guaranteed: a non-increasing bubble as the budget grows. Both candidate sets
  are built from the budget, so a schedule found at a smaller budget is not regenerated at a larger one. Counterexample
  (costs 1/1/1, `mem_f` 1, `mem_b` −0.5, comm 1.0, 4 stages × 32): 16.52% at budget 5.95, 17.95% at 6.1.
- **SCHEDULER-I14.** (DEFERRED-TO: M3) ZB-V: for two chunks per rank, the solver produces V-shaped orders with the paper's bubble and
  memory bounds, validated by an extended simulator.
- **SCHEDULER-I15.** A between-step re-solve from runtime telemetry (new peer weights and ownership) produces a valid
  order. The fold order is unchanged, so the result stays bit-identical (with SPEC-runtime task 2).
  The measured hop term is the sender's seconds per **transfer** (one boundary tensor): per summary and hop entry,
  `busy_ms / 1e3 / n̂`, then the median over the window, in chain order; the link's one-way latency is added by
  `hop_comm_seconds` (busy time ends when the bytes leave the sender; `DelayedDelivery` delays at the receiver).
  `n̂ = frames` when `spec.p2p_streams == 1` (one `send` event per transfer); otherwise each event is one fragment,
  so `n̂ = bytes / _boundary_bytes(spec)` (the coded bytes per transfer the solver already models). Never
  `busy_ms / frames` with `p2p_streams > 1`: that is seconds per fragment, ⌈payload / 1 MiB⌉ times too small.
  Test: `test_hops_from_telemetry_counts_transfers_not_fragments` (`p2p_streams` 2, 8 fragments per transfer).

- **SCHEDULER-I16.** (DEFERRED-TO: M2b) The planner admits `shard: "pieces"` at width w for a stage, with its peers
  placed on sites, only when five conditions hold. Otherwise it refuses that width and names the number, link or class
  that refused:
  - every peer shares the reference's device class: the same device type and device model (`gpu_name`), checked at
    admission and at every re-plan;
  - S1 holds on every link ℓ (each NIC and each PCIe path per direction, each host's DRAM, each site uplink per
    direction), summed over every stage s and every GPU g that loads ℓ in the proposed placement, hosts included:
    `Σ_{(s,g) on ℓ} k_ℓ·(b_g·P_s·share_{s,ℓ} + A_{s,ℓ})/c_s ≤ cap_ℓ`, with `c_s` the stage's compute per
    microbatch as in S2 and `b_g·P_s = Σ_u b_g,u·P_u` over the stage's units (the Embed's
    fp32 tables at 4 bytes); `share = (w−1)/w` on a NIC or PCIe path per direction, `n_a·n_b/w` on an uplink between
    sites holding `n_a` and `n_b` of the stage's peers, and `2·(w−1)/w` on host DRAM (both directions); `A_{s,ℓ}` the
    stage's pipeline-boundary bytes per microbatch round on ℓ (on host DRAM, both directions); and `k_ℓ` the passes per
    sent byte per side: 1 on a NIC, PCIe path or uplink, and on host DRAM the transport's counted passes per side,
    which apply to exchange and boundary bytes alike (15 today on the sealed M2 data plane, 17 when adaptive microbatching fragments the exchange frames (`adapt_budget_ms` > 0): the mean of the sender's
    12 and the receiver's 18 passes on the worst default path of either frame kind, since S1's factor 2 sums both
    sides; counted by SC-5, `docs/research/adr-0006/sc5-passes.md`, by the method of ADR-0006 `round3.md` §2 and
    `round4.md` §5, which counted 10 before sealing; at 10, a 600 GB/s host admits no GPU of point B's stage 0,
    `docs/experiments/m2b-e7/PLAN.md`, and at 15 no GPU of point B's stage 0 or interior stages either);
  - S2 holds: `E − window_s ≤ ε·C`, with `E` and `window_s` from SCHEDULER-I17 and `C = (M/w)·max_s c_s`, the
    pipeline's compute per step at its slowest stage. `c_s` is one stage's F + B + W per microbatch from the
    planner's own costs (`layer_costs` through `zero_bubble.analytic_costs`, attention included, recompute as the
    recipe runs; the Embed's tables carry none), the same value in S1, S2 and C (ADR-0006 R16, the human's ruling of
    2026-10-09; DESIGN §9's `8·P^F_s·T_mb/F` is the model's approximation of it);
  - device memory fits for every peer k: `b_p·P_s + 16·P_s/w + b_g·M_k·P_s/w + b_g·P_module,max +
    min(S, M/w)·T_mb·h·b_boundary + L_s·T_mb·16·h·4`, plus the head's logits on the last stage, against the device less
    `vram_reserve`, with `M_k` peer k's owned microbatches (the own-piece hold of RUNTIME-I13);
  - host memory fits on every host: the sum, over every peer k placed on that host (of any stage), of its one-step
    bound `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w` (RUNTIME-I13; `b_p·P_s = Σ_u b_p,u·P_u`) plus its
    pinned copy chunks in flight, against the host's memory less its reserve; otherwise the refusal is `MEM-HOST`,
    naming the host. A re-plan that changes `M_k` or the placement re-checks both memory conditions.

  `cap_ℓ` is a capacity measured or probed under the concurrent load, never one flow's rate at another width. A peer's
  site comes from its declared region (CONTROL-I11). F is the profiled rate with the recipe's deterministic kernels, and
  `T_mb` is the recipe's. ε is the human's bound.
- **SCHEDULER-I17.** (DEFERRED-TO: M2b) The slot-level simulation models each sharded stage's exchange per unit
  (`SplitLinear` module, and the Embed's tables as one fp32 unit released when W(mb) starts, needed first by F) as the
  runtime sends it: one send queue per (sender, destination), served need-first (forward
  unit, gathers before gradients) one chunk of at most 16 MiB at a time (RUNTIME-I13); each peer's egress and ingress,
  each host's DRAM and each site uplink, per direction, shared max-min fairly among the chunks in flight; a peer's
  egress and ingress at `min(NIC, PCIe)` with `2·f/PCIe` per hop. A W releases a unit's gradient pieces when it computes the unit, in the
  stage's emission order (MODEL-I7); a unit's gather is released after the owner's last fold of it and its AdamW, at
  their HBM op times (`(b_g + 16)` B per folded element for a 2-byte wire, 12 for fp32; 30 B per updated element);
  step t+1's F on a unit waits for that unit's gather. The fold of every microbatch is device time in the step:
  `M·(P_s/w)` elements per peer. Op times are per peer (`simulate_slots`). The predicted step time, `E` and `window_s`
  come from that one simulation, and the plan takes the emission order with the smaller predicted step time. With the
  link-bound period it gives `E = max(X_s, V/cap + RTT + t_gap − C)` (ADR-0006, derivation §2).
- **SCHEDULER-I18.** (DEFERRED-TO: M4; ADR-0007) At a first launch (no wave exists):
  - with `build: "slice"`, selection applies CONTROL-I19's exclusion rule: only nodes whose sampler fingerprint (HELLO)
    equals the init record's, and that no slot report excluded, are eligible. Like `same_region`, selection picks the
    best fleet within that group; it refuses with `INIT-CLASS` and the count of nodes per fingerprint only when the
    group cannot field the plan. The fingerprint is never taken from the selection;
  - the build and every later resume fit on every host: the sum over the host's slots of the larger of the build
    charge and the resume charge, plus the interpreter's baseline per slot, against the host's memory less its
    reserve; otherwise the refusal is `MEM-INIT`, naming the host and the term that binds. The build charge is
    `4·(P_layer,max + P_held)` (`build: "slice"`, MODEL-I6) or `4·P` (`build: "full"`, `arch="user"`); the resume
    charge is the slot's wave as `_resume` loads it (RUNTIME-I14), `12·P_s` for a whole-tensor AdamW slot; each adds `4·P_s` for a slot whose device is
    the CPU. A resume excludes no node from init; the replacement of a CPU-device slot in a run with a record must
    have the record's fingerprint (CONTROL-I19, ADR-0007 R9).
- **SCHEDULER-I19.** (DEFERRED-TO: M2b) The engine bridge gives each peer of a wide stage a lane-consistent order solved
  at that peer's own costs and activation budget, and keeps the stage-level split only where `simulate_slots` finds it
  faster among memory-feasible orders; width-1 orders are unchanged (ADR-0006 R14: 43.8% idle against 109.4% at S 8,
  w 4, M 32, derivation §5).

- **SCHEDULER-I21.** (DEFERRED-TO: M4; ADR-0008) For a stage holding MoE blocks, compute is charged at active and
  bytes and memory at total parameters.
  - `layer_costs`, the analytic F/B/W costs and SCHEDULER-I16/I17's `c_s = 8·P^F_s·T_mb/F` count, per MoE block, its
    attention, router, k experts and shared expert; `P_s` in S1's `b_g·P_s`, the working copy `b_p·P_s`, the state
    `16·P_s/w` and every memory and host bound counts all E experts.
  - The pending-W term uses SPEC-model §2's MoE count.
  - SCHEDULER-I17 models an MoE block's experts of one projection as one unit (SPEC-runtime task 5), released when
    the W computing its last expert completes, and adds the selection biases' int64 count lane (`L_m,s·E·8` bytes per
    microbatch per peer, RUNTIME-I18) to the small-tensor unit's gradient bytes and the biases to its gathers. Under
    HF modules (ADR-0009) the two 3-D expert units are produced in B: released when W(mb) starts, like the Embed's
    tables, at the master dtype's bytes (4 under today's recipe), and their dW is charged to B, not W (ADR-0008
    §Mathematical Model, HF v5 table). Their gradients, `b_m·P_bank,s` for the stage's 3-D expert elements, are held
    from B's end to W's start, whole on every peer of a wide stage: `stage_memory_model` charges them in `mem_b` and
    frees them in `mem_w` (§2.1), so the solve's replay holds `m_BW·b_m·P_bank,s` while `m_BW` microbatches are in
    flight, and SCHEDULER-I16's device-memory condition adds `m_BW,k·b_m·P_bank,s`, with `m_BW,k` the most of peer
    k's microbatches between B's end and W's start in the emitted order (`M_k` before an order is solved). A plan
    whose order exceeds a budget with it is refused like any memory overrun.
  - `c_s` in S1 and S2 stays the analytic FLOP count, a lower bound of an MoE stage's compute (it omits per-expert
    launches and host count reads), so S1 and S2 are conservative for MoE stages: they may refuse a feasible width,
    never admit an infeasible one.
  - A plan's rationale names ρ_s = P_s/P^F_s for every stage it refuses pieces for, and records each MoE stage's
    analytic `c_s` next to its profiled per-microbatch time.
- **SCHEDULER-I22.** (DEFERRED-TO: M4; ADR-0009) Plans of HF models charge what the runtime moves and holds.
  - `bytes_per_elem` at every cut of an `arch="hf"` model is the bytes of HF's residual dtype under the recipe (4 under
    fp32 parameters with autocast); a declared narrower boundary is refused naming it.
  - For a tied table whose prologue and epilogue units fall on different stages, the Head stage's and stage 0's memory
    count §2's tied terms, every chain hop between them carries §2's tied bytes in the solve's per-hop communication
    (and in S1 for sharded plans), and the step adds `ΔT_tied`. No link outside the chain is planned or probed. Such
    a plan with stage 0 or the Head stage at width > 1 is refused, naming it (RUNTIME-I19; ADR-0009 R14, open).
  - The activation memory of a stage of eager-attention layers includes `b·H·S²·s` B per layer (§2).
  - The rationale names the boundary dtype, the tied hops and `ΔT_tied` with its five terms.

- **SCHEDULER-I23.** `scheduler/shape.py` prices a dense shape's WAN bubble from measured figures: compute per
  microbatch per stage 8 x stage parameters x seq / effective FLOP rate, transfer seq x d x 2 x wire ratio / hop rate,
  bubble 1 - (compute / max(compute, transfer)) x (1 - fill), fill (S - 1) / (M + S - 1). `advise()` returns the
  largest standard shape (d a multiple of 128, 128-dim heads, d / n_layer within 20-200) whose heaviest stage fits the
  memory budget and whose priced bubble meets the target, else the least-bubble fitting shape. It reproduces the
  2026-10-12 H100 run (priced 78%, measured 72-79%).
- **SCHEDULER-I24.** The partition DP prices a stage's memory feasibility from the measured fit of its node when it
  has one (`_stage_fits`: the fit's peak minus its entries' state, the rest scaled per entry), else from the analytic
  model with the run optimizer's state bytes per parameter (`adamw-host` keeps no moments on the device). A cut is
  refined only to one every stage fits (`refined_partition`).


## 5. Ordered task list

1. **M2 — degrade instead of refuse (SCHEDULER-I13).** Today `zero_bubble(cfg, "given")` raises
   `no zero-bubble schedule fits the memory limit`. Example: 7B on H100, 200 ms one-way, 8 GB budget (the test fixture
   `_seven_b`: 8 stages, 1024 microbatches): 17.31% degraded at 8 GB, 17.24% from the grid at 16 GB.
   - Search the warm-up family of §2.1 (gaps over the largest feasible warm-ups, shortest simulated makespan), take the
     better of it and the fitting grid candidates, and accept the larger bubble. The largest warm-ups themselves are
     the family's caps, not its answer.
   - Report the bubble and the warm-ups in `params` and the rationale.
   - The bubble's monotonicity in the budget is a regression check on pinned grids (the 4-stage grids, the
     no-recompute comm-2.0 grid and the 7B budgets), never asserted as a general property (SCHEDULER-I13).
   - `solve()` uses the measured budget as given (no 1F1B floor); refuse only below one microbatch, naming the bytes.
2. **M2 — re-plan from telemetry (SCHEDULER-I15).** `schedule_for_spec` accepts measured per-op times from runtime
   `TELEMETRY`. The coordinator re-solves when the measured makespan deviates more than X% from the prediction (X decided
   in the M2 handout).
   - `replan(spec, windows, min_gain=0.10)`; the threshold is A7's rule (HANDOUT-M2 §8 A7: SPEC-control task 2 and
     CONTROL-I21).
3. **M3 — ZB-V (SCHEDULER-I14).**
   - Extend `ZBConfig` with two chunks per rank.
   - Port the reference's V schedule (`zbv.py`), with per-stage costs and per-hop comm.
   - Simulate with chunk-aware dependencies: chunk 1's forward runs from the last rank back to the first.
   - Make `solve_microbatches` chunk-aware.
4. **M2b — plan mode for wide and sharded stages (SCHEDULER-I16, I17; ADR-0006).** `plan_job` emits
   `stage_widths > 1` with `shard: "pieces"` only where SCHEDULER-I16 admits it. The candidates are: the deepest cut
   at w = 1, on fewer devices when there are more devices than cut points; narrower stages; wide stages with
   `shard: "zero1"` (today's layout and step barrier, with the job's `wire`), which keep today's admission (device
   memory) and are priced by the same simulation with their own owners, host DRAM and boundary bytes
   included; and `shard: "pieces"` inside a site or across a link where SCHEDULER-I16 admits it. It picks the
   candidate with the smallest simulated step time, compares placements by simulated step time, never by bytes or
   FLOPs per byte alone, and refuses the job only when no candidate fits memory, as today. The rationale names every
   refused width with the link, number or class that refused it.
   - Depends on the engine bridge's lane-order fix (SCHEDULER-I19; ADR-0006 R14, M2b's first task by the human's decision): today's stage-level solve at `1/Σ(1/c_peer)` followed by `slot_orders` idles 109.4% at
     S 8, w 4, M 32 against a lane solve's 43.8% (ADR-0006, derivation §5), so widths compared through it are biased
     against width.

5. **M4 — MoE costs (SCHEDULER-I21, ADR-0008).** `layer_costs`, `analytic_costs` and `stage_memory_model` read
   the config's MoE fields; SCHEDULER-I16/I17's terms take active and total parameters as I21 states (on M2b's merged
   code).

6. **M4 — HF models and tied tables (SCHEDULER-I22, ADR-0009).** `layer_costs` and `stage_memory_model` from the HF
   config and skeleton shapes; the boundary dtype from the recipe (any other declared one refused); the tied bytes in
   the per-hop communication, `ΔT_tied` in the step and the tied memory of the Head stage and stage 0; the eager
   attention term; `planner.py:160`'s once-counted table replaced by I22's charges. Rollback (ADR-0009 `round6.md`
   §4): this task reverts with runtime 7a/7b to "tied HF models at S > 1 refused at planning"; the boundary-bytes
   and eager terms revert with SPEC-model 4b.

## 6. Decisions

- **The solved order is the only execution model.** Measured on CPU with 4 stages, s/step from 5 to 40 ms one-way:
  - old 1F1B: 0.634 / 1.123 / 1.485;
  - profiled zero bubble: 0.732 / 0.924 / 1.067.

  Rejected: keeping 1F1B as a fallback.
- **Static IWRR over dynamic stealing.** A dynamic pool cannot be solved ahead, and the order is what makes timing
  predictable.
- **Per-stage costs instead of one median cost.** Heterogeneous fleets are the target. A median cost hides the warm-up
  overrun that the clamp in §2.1 fixes.
- **Selection by measured makespan, not the reference's formula.** Comm and heterogeneity make the closed form wrong.
