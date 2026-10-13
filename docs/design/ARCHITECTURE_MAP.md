# Architecture map: `wantrain`

A file-by-file map of the engine for agents that must find code before they change it. It complements
[CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md), which gives each component's boundaries. Symbols are the public
API as it exists on `main`. Line counts are approximate. Workstream keys refer to the `DESIGN.md` workstreams block.

## Entry points

| File | Responsibility | Key symbols |
|---|---|---|
| `src/wantrain/cli.py` (270) | `wantrain` command: `coordinator`, `worker`, `stage`, `local`, `plan`, `compare`, `validate-model`, `prepare-data` | `main`, `cmd_coordinator`, `cmd_worker`, `cmd_local`, `cmd_plan` |
| `src/wantrain/worker/stage_main.py` (105) | One pipeline slot as an OS process. It prints the `WT_*` markers the agent parses. | `run_stage(spec_path, rank, peer)`, `stream_losses` |
| `src/wantrain/launch/local.py` (150) | Every slot of a spec on this machine, plus the oracle and the diff | `run_local`, `run_and_compare`, `ensure_reference`, `links_for`, `recipe_digest` |
| `examples/wan_cluster_demo.py` | 5 workers, 2 emulated regions, a worker killed mid-run, heal, bit-exact result | script |

## runtime (workstream `runtime`)

| File | Responsibility | Key symbols |
|---|---|---|
| `runtime/spec.py` (200) | `JobSpec`, the single JSON object that determines a run. `finalize()` fills the partition, links, widths, IWRR ownership and the analytic zero-bubble order, then validates them. | `JobSpec.finalize`, `.reschedule`, `.owned(rank, peer)`, `.device_for`, `.to_json/.from_json/.load`, `LinkCfg`, `OPS = ("F","B","W")` |
| `runtime/stage.py` (1180) | `StageRuntime`: executes `schedule_order[rank][peer]` (F, B, W, plus runnable-W fill). Also chain/mesh connection, sender/reader threads, fragments, adaptive microbatching, ZeRO-1 in wide stages, checkpoint and resume, telemetry, and the END shutdown handshake. | `StageRuntime(spec, rank, peer).run()`; internals `_run_step`, `_forward`, `_run_backward`, `_run_w`, `_fill_w`, `_optimizer_step_local/_wide`, `_checkpoint`, `_resume`, `_connect_chain`, `_connect_mesh`, `_wire_multi`, `_sender_loop`, `_reader`, `_finish`; `compute_autocast`, `engine_supplies_autocast` |
| `runtime/accumulate.py` (75) | Folds gradients in a fixed order (microbatch index), whatever the arrival order. Has an audit quarantine. | `OrderedAccumulator.add/release/reject/complete/result` |
| `runtime/determinism.py` (40) | Seeds, deterministic algorithms, no TF32, cuDNN benchmark off, x86 ISA pin; exact float hex | `setup_determinism`, `float_hex` |
| `validation/reference.py` (170) | The oracle: a single-device run of the same recipe with the same microbatch order | `run_baseline(spec, tag, ...)` |
| `validation/compare.py` (55) | Exact diff of two loss sequences (hex) | `compare(baseline, candidate)` |
| `validation/model_contract.py` (345) | Checks a user `build(cfg)` model: flat list, forward, determinism, statelessness, no dead layers | `validate(source, target)`, `check_*` |

## model (workstream `model`)

| File | Responsibility | Key symbols |
|---|---|---|
| `model/gpt.py` (450) | Built-in decoder as a flat list `[Embed, Block*n, Head]`; config, presets, boundary narrowing, loss. It always converts linears to `SplitLinear`. | `GPTConfig`, `make_config`, `build_layers(cfg, seed)`, `loss_fn`, `narrow`, `resolve_sdpa_backend` |
| `model/arch.py` (235) | Llama-family pieces: RMSNorm, SwiGLU, RoPE, generalized Embed/Block/Head | `build_layers`, `make_norm`, `make_mlp`, `rope_cache`, `apply_rope`, `gpt2_init` |
| `model/wgrad.py` (155) | Deferred weight gradients, the zero-bubble B/W split. B captures `(X, dY)` per linear; W computes `dW = dYᵀX`. The store is process-wide, because CUDA and MPS run backward on autograd device threads. | `capture()`, `flush(store)`, `SplitLinear`, `convert(module)`, `owned_params` |
| `model/precision.py` (160) | Declared precision recipes (compute and boundary dtypes) and whether each is lossless | `spec`, `classify`, `bytes_per_elem`, `compute_runnable`, `describe` |
| `model/fp8.py` (95) | Recipe-declared fp8 boundary with per-token power-of-two scales | `quantize`, `dequantize`, `roundtrip`, `scale_exp`, `recover_exp` |
| `model/lora.py` (150) | LoRA over any built architecture; frozen base | `apply`, `LoRALinear`, `split_params` |
| `model/hf.py` (360) | Strict loading of HuggingFace decoder checkpoints into the flat layer list | `config_from_hf`, `load_into`, `checkpoint_digest`, `LoadReport` |
| `model/data.py` (65), `model/dataset.py` (265) | Deterministic batches; corpus preparation, manifests and shards | `make_batch`, `prepare`, `shard_view`, `check_compatible` |

## scheduler (workstream `scheduler`)

| File | Responsibility | Key symbols |
|---|---|---|
| `scheduler/zero_bubble.py` (610) | Zero-bubble heuristic (port of sail-sg `zb.py`, per-stage costs and per-hop comm), stage and slot simulators, IWRR, and the engine bridge | `ZBConfig`, `Schedule`, `zero_bubble(cfg, mem_mode)`, `simulate`, `simulate_slots`, `bubble_rate`, `iwrr`, `stage_memory_model`, `hop_comm_seconds`, `analytic_costs`, `slot_orders`, `solve`, `default_schedule`, `schedule_for_spec` |
| `scheduler/schedule.py` (245) | Microbatches per step from the solved schedule's bubble; fleet ceiling; in-flight depth | `solve_microbatches`, `fbw_costs`, `solve_fleet_ceiling`, `solve_k_inflight`, `stage_live_graphs` |
| `scheduler/partition.py` (190) | Exact min-max DP over contiguous layer ranges with per-layer costs and a memory predicate; chain ordering | `assign_layers`, `layer_costs`, `proportional_partition`, `order_stages`, `best_chain` |
| `scheduler/placement.py` (110) | Binds a recipe to measured nodes: peer weights, cut, links, `reschedule()` | `place`, `stage_speeds`, `memory_order`, `rank_memory_weights`, `emulated_links` |
| `scheduler/planner.py` (495) | Plan mode: from a measured fleet to a complete JobSpec (precision, mb size, microbatches, partition) | `plan_job`, `Fleet`, `Node`, `Overrides`, `stage_fits`, `stage_param_split`, `largest_mb_size` |
| `scheduler/autoconfig.py` (360) | Boundary-precision ladder priced against the link (overlap condition) and memory | `evaluate`, `choose`, `plan`, `floor_mbps`, `tokens_per_s_ceiling`, `boundary_bytes_per_token`, `stage_memory_bytes` |
| `scheduler/selection.py` (320) | WAN-aware node selection: one region, distinct addresses, link memory, capability window | `select_nodes`, `candidate_fleets`, `SelectionPolicy`, `Selection` |
| `scheduler/linkmemory.py` (175) | Link verdicts and dead hosts remembered across runs | `LinkMemory`, `PathRecord`, `verdict`, `path_key` |
| `scheduler/fleetplan.py` (205) | Fleet JSON → JobSpec (endpoints, devices) | `build_spec`, `model_config`, `PlanRefused` |
| `scheduler/production.py` (45) | The always-on production flags | `PRODUCTION`, `spec_fields`, `deviations` |

## transport (workstream `transport`)

| File | Responsibility | Key symbols |
|---|---|---|
| `transport/framing.py` (165) | Data-plane frames (`FWD, BWD, END, GRADS, PARAMS, LOSS, HELLO`), fragments, payload limit | `send_frame`, `recv_frame`, `send_end`, `Reassembler`, `Frame`, `MAX_PAYLOAD` |
| `transport/serial.py` (170) | Exact, code-free object codec (no pickle) | `dumps`, `loads`, `SerialError` |
| `transport/identity.py` (165) | ed25519 peer identity and allowlist; HMAC fallback with a required token | `authenticate`, `load_or_create_key`, `peer_id` |
| `transport/control.py` (145) | Coordinator ↔ agent JSON channel, file streaming, token MAC | `Channel`, `token_mac`, `check_mac`, `ChannelClosed` |
| `transport/shim.py` (80) | Per-link WAN emulation (latency, bandwidth, jitter) | `LinkSpec`, `SenderPacer`, `DelayedDelivery` |
| `codec/planes.py` (190), `codec/f32_planes.py`, `codec/fp8_planes.py` | Lossless boundary wire codecs (byte planes + zstd) | `encode`, `decode`, `encode_scaled_fp8` |

## control (workstream `control`)

| File | Responsibility | Key symbols |
|---|---|---|
| `coordinator/coordinator.py` (1210) | Control plane: registry, selection, probing and routing, planning, F/B/W profiling, cut refinement, zero-bubble solve, launch, monitor, heal, restart | `Coordinator.run_job/start/stop/wait_for_workers`, `CoordinatorConfig`, `Job`, `RunResult`, `Worker`; internals `_setup`, `_profile`, `_refine_partition`, `_zero_bubble`, `_monitor`, `_heal`, `_restart_in_place` |
| `worker/agent.py` (650) | Node daemon: HELLO and measured capabilities, heartbeats; ASSIGN/PROFILE/FIT/MEASURE/PROBE/ABORT/RESTORE/RELEASE; supervises the stage process; ships checkpoint waves | `WorkerAgent.run`, `StageProcess` |
| `fault_tolerance/liveness.py` (275) | Judge: ok / suspect / convict / stalled / failed / done | `LivenessJudge.judge`, `SlotState`, `Verdict` |
| `fault_tolerance/checkpoint.py` (155) | Waves per `(rank, peer)`, common step, prune, sha256, resume spec | `CheckpointStore`, `patch_spec_for_resume`, `sha256_file` |
| `fault_tolerance/replacement.py` (95) | Chooses the spare for a dead slot | `find_replacement` |
| `fault_tolerance/audit.py` (125) | Verify-before-fold policy (library, not wired into the runtime) | `AuditPolicy`, `digest`, `Reputation` |
| `resources/probe.py` (165) | Capability probe: matmul TFLOPs, memory, device (CUDA/MPS/CPU) | `measure`, `bench_tflops`, `best_device` |
| `resources/links.py` (300) | Link probes, bidirectional measurement, duplex classification, emulation | `measure_pair`, `measure_row`, `classify_duplex`, `Emulation`, `LinkProfile` |
| `resources/profile.py` (170) | Times F, B and W of one slot on its own device (`WT_PROFILE`) | `profile_stage` |
| `resources/fit.py` (165) | Measured peak memory of one slot under its schedule (`WT_FIT`) | `measure`, `peak_pending` |
| `resources/node.py` (105) | What the scheduler knows about a node | `NodeInfo` |

## Tests

| Path | What it holds |
|---|---|
| `tests/unit/test_zero_bubble.py` | Paper formulas, reference golden orders, heterogeneous validity, simulator deadlock rejection, deferred-dW bit-exactness incl. device threads, IWRR, slot orders |
| `tests/unit/test_schedule.py`, `test_scheduler.py` | Microbatch solver, fleet ceiling, partition DP vs brute force, routing, selection, placement, planner |
| `tests/unit/test_transport.py`, `test_codecs_identity.py` | Serial codec, frames, fragments, END, shim; codecs, ed25519, allowlist, HMAC |
| `tests/unit/test_fault_tolerance.py`, `test_audit.py`, `test_resources.py` | Liveness rules, checkpoint store, replacement; audit policy; probes and links |
| `tests/unit/test_finetune.py`, `test_user_model_precision.py` | HF loading, LoRA; user-model autocast |
| `tests/integration/test_engine_bitwise.py` | Every mechanism as a real multi-process run, diffed bit-for-bit against the oracle |
| `tests/integration/test_coordinator.py` | Coordinator end-to-end: lifecycle, heterogeneity, WAN routing, planner, CPU + MPS fleet, wide stage |
| `tests/chaos/test_recovery.py` | Worker or stage death, freeze, control-plane blip, no spare, wide-stage peer death (harness-owned) |
| `tests/helpers/cluster.py` | Local cluster harness: coordinator in-process, agents as subprocesses, fault injection |
