# Migration from Elysium: audit and decisions

Source: `virgiliolrf/Efficient-Distributed-Training-with-Heterogeneous-Hardware-over-Wide-Area-Networks`.
GitHub `main` holds 2 commits (`7aed4f3`); the engine's later evolution (74
commits plus uncommitted work, including the panel's fault-tolerance logic)
exists only in the local clone. The audit read all three states; the migration
takes the **latest working tree** of the engine, and the GitHub `main` version
was used to bisect a regression (below).

Rule applied to every component: *is it required to execute, coordinate,
schedule, communicate, synchronize, recover, or efficiently run distributed
training over a WAN?* Yes → migrate or adapt. No → discard.

## Classification

KEEP = moved as is (imports only). MIGRATE = moved with small, local changes.
ADAPT = rewritten or generalised to remove product coupling. DISCARD = not moved.

### Engine (`pipeline/src/elysium/`)

| Source | Decision | Target | Notes |
|---|---|---|---|
| `engine/accumulate.py` | KEEP | `runtime/accumulate.py` | fixed-order fold, audit quarantine |
| `engine/spec.py` | ADAPT | `runtime/spec.py` | + `schedule_order`, `peer_weights`/`mb_owner`, per-slot `devices`, `trusted_peers`, `dial_timeout_s`; − schedule switches (see Unification) |
| `engine/stage.py` | ADAPT | `runtime/stage.py` | one execution model (zero-bubble order per slot), shutdown handshake, safe frames, per-slot device, loss history in shards, `WT_STEP` progress (see Unification) |
| `transport/framing.py` | ADAPT | `transport/framing.py` | pickle → `OBJ` codec, `END`, payload limit, errors instead of asserts |
| `transport/shim.py` | KEEP | `transport/shim.py` | WAN emulation per link |
| `transport/identity.py` | ADAPT | `transport/identity.py` | no default token, peer allowlist, atomic 0600 key |
| — | new | `transport/serial.py`, `transport/control.py` | exact, code-free object codec; control channel |
| `codec/*` | KEEP | `codec/*` | lossless wire codecs |
| `model/*` | KEEP | `model/*` | arch name `elysium` → `builtin`; HF hub revision pinning (`name@rev`) |
| `utils/determinism.py` | MIGRATE | `runtime/determinism.py` | x86-only ISA pin |
| `control/partition.py` | MIGRATE | `scheduler/partition.py` | + branch & bound, greedy beyond 8, `best_chain` (from `mesh.ts`), `proportional_partition` |
| `control/plan.py` | MIGRATE | split: `scheduler/partition.py`, `resources/probe.py` | |
| `control/schedule.py` | MIGRATE | `scheduler/schedule.py` | microbatch solver now prices the solved zero-bubble schedule instead of the 1F1B fill/drain |
| `control/autoconfig.py` | KEEP | `scheduler/autoconfig.py` | precision ladder, memory model, admission |
| `control/production.py` | KEEP | `scheduler/production.py` | production flag set |
| `control/provision.py` | MIGRATE | `scheduler/planner.py` (`plan_job`) | + per-node device; reference-window guard |
| `control/master.py` | ADAPT | `coordinator/coordinator.py`, `scheduler/placement.py` | one-shot master → long-lived coordinator with fault tolerance |
| `control/agent.py` | ADAPT | `worker/agent.py` | in-process stage → supervised child, reconnects, ships waves |
| `control/prober.py` | ADAPT | `resources/links.py` | + reverse, bidirectional, duplex, emulation |
| `control/admission.py` | DISCARD | — | superseded by `autoconfig.floor_mbps` (source deleted it too) |
| `control/validate.py` | MIGRATE | `validation/model_contract.py` | |
| `control/compare.py` | KEEP | `validation/compare.py` | |
| `control/audit.py` | MIGRATE | `fault_tolerance/audit.py` | library; not yet wired into the runtime |
| `pipeline/tests/*` | MIGRATE | `tests/unit/*` | 49 tests, all passing |

### Entry points and harnesses

| Source | Decision | Target |
|---|---|---|
| `train_baseline.py` | MIGRATE | `validation/reference.py` (the oracle) |
| `start_fleet.py --plan` | ADAPT | `scheduler/fleetplan.py` (snake_case fleet; panel camelCase shim dropped) |
| `start_fleet.py --stage` | ADAPT | `worker/stage_main.py` (TensorBoard writer dropped) |
| `start_fleet.py --prepare` | ADAPT | `wantrain prepare-data` |
| `fit_check.py` | MIGRATE | `resources/fit.py` |
| `probe_node.py` | ADAPT | `resources/probe.py` (CUDA-only → CUDA/MPS/CPU) |
| `start_master.py`, `start_worker.py`, `start_stage.py` | ADAPT | `wantrain coordinator / worker / stage` |
| `experiments/phase0/run_local.py`, `run_all.sh` | ADAPT | `launch/local.py`, `tests/integration/test_engine_bitwise.py` |
| `setup_and_launch_*.sh` | DISCARD | Tailscale/Docker-specific launchers |
| `experiments/phase1..6`, `pretrain_math`, `optimizations` | DISCARD | one-off paper experiment drivers |
| `analysis/*` | DISCARD | research reports; the formulas live in `scheduler/schedule.py` and `autoconfig.py`. `BUSINESS_MODEL.md`, `economics.py`: business |
| `docs/wan-viability.md` | DISCARD (folded) | findings live in the admission/planner code and ARCHITECTURE.md |
| `deploy/Dockerfile`, `deploy/provision_vast.md` | DISCARD | Vast.ai image/provisioning |

### Panel (`panel/`, TypeScript/Next.js)

Product layer, discarded as a whole. Its orchestrator, however, held the only
fault-tolerance and WAN-aware selection logic in the source; those
*algorithms* were ported to Python without the marketplace, SSH and billing:

| Source | Decision | Target |
|---|---|---|
| `orchestrator/liveness.ts` | ADAPT | `fault_tolerance/liveness.py` (SSH probes → agent heartbeats/exit reports; + reconnect grace, stall → restart in place) |
| `orchestrator/heal.ts` | ADAPT | `fault_tolerance/checkpoint.py` (waves per `(rank, peer)`, prune by common wave), `replacement.py` |
| `orchestrator/replace.ts` | ADAPT | `Coordinator._heal`, setup swaps (renting → spare pool) |
| `orchestrator/rendezvous.ts` | ADAPT | `Coordinator._route_chain`, `_plan` (plan + fit + re-plan), `_restore_and_relaunch`, last-rank-first launch |
| `orchestrator/select.ts` | ADAPT | `scheduler/selection.py` (price ranking → measured capacity; heterogeneous `any` grouping added) |
| `orchestrator/linkmemory.ts` | ADAPT | `scheduler/linkmemory.py` |
| `orchestrator/mesh.ts` | ADAPT | `scheduler/partition.best_chain` + coordinator probing |
| `orchestrator/link.ts` | ADAPT | `resources/links.py` (iperf3 → in-process bidirectional probe) |
| `orchestrator/runner.ts` | ADAPT | `Coordinator._monitor` (exit judging via liveness classes) |
| `orchestrator/poll.ts` | DISCARD | Vast image-pull stall telemetry |
| `orchestrator/{ssh,code,provision,access,pubkey,tbproxy,diagnostics,metrics,advice,precision,store,types}.ts` | DISCARD | SSH transport, Vast provisioning, dashboard/UI, run store |
| `vast/*`, `src/app/api/*`, `public/ui/*`, `cli/elysium`, `gateway/*`, `auth.ts`, `middleware.ts`, `workspace.ts`, `zip.ts` | DISCARD | marketplace, web app, auth, UI, product CLI |
| `.runs/` | DISCARD | run artifacts |

## Defects found in the source and fixed during the migration

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | A normal shutdown killed slower ranks: senders never sent the `END` frame readers expect, and local HEAD made every reader EOF fatal | phase-0 `m3` (3 stages, 20 ms + 40 ms jitter) failed 3/3 on HEAD and the working tree, passed on GitHub `main` | senders send `END` and half-close after draining; readers keep draining until every neighbour's `END`; then close |
| 2 | Ring-mode sender threads were not joined at shutdown (last gradient could be cut) | `_wire` did not register its thread | registered and joined |
| 3 | A failing dial or p2p handshake inside a helper thread was swallowed (`None` socket, crash later) | `_Result`, `_auth` | errors re-raised on the main thread; optional dial timeout |
| 4 | A resumed run lost the loss history before the resume point | `_resume` restored weights/optimizer only | loss history travels in the shard |
| 5 | `pickle` on network sockets (control and data planes): remote code execution for anyone reaching a port | `master.py`, `framing.py` | `transport/serial.py` (exact tensors, tagged JSON) |
| 6 | HMAC fallback authenticated with the hard-coded token `"elysium"` | `identity.py` | token required; ed25519 allowlist option |
| 7 | Peer key written world-readable and racily by concurrent stages | `identity.py` | atomic create, mode 0600 |
| 8 | Unbounded frame length from the network | `framing.py` | `MAX_PAYLOAD` |
| 9 | `ATEN_CPU_CAPABILITY=avx2` forced on ARM (warning, no effect) | determinism | x86 only |
| 10 | Planner crashed when a reference window was given without its layer count | mypy | explicit error |

## Dependencies

Added: `zstandard`, `cryptography` (optional with fallbacks in the source, now
core: zstd is ~70× faster than zlib-6 on boundary planes, ed25519 is the data-plane
identity), extras `data` (`tiktoken`) and `hf` (`safetensors`), dev group
(`pytest`, `ruff`, `mypy`), build backend `hatchling`. `torch==2.13.0` deploy pin
→ `torch>=2.4` (bitwise claims are per declared device map and stack; pin in
the deployment, not the library).

Removed: the whole Node/Next.js stack of the panel, the Vast.ai API client,
SSH/rsync transport, `iperf3` (replaced by in-process probes), TensorBoard,
Tailscale-specific launch scripts, the Vast Docker image.

## Unification: zero-bubble pipeline parallelism as the only path

The source engine carried every milestone as a switch: monolithic 1F1B, a
two-call B/W split, a greedy event-driven "deferred-W battery", interleaved
1F1B, fifo/lifo/random policies, a ring with a wrap link, and a dynamic claim
pool for wide stages. The engine now has one execution model:

| Removed | Replaced by |
|---|---|
| monolithic backward (`split_backward=False`) and the two-call split | `model/wgrad.py`: B computes dX and defers each linear's dW; W runs the dW GEMMs (the reference implementation's WeightGradStore). The two-call split re-walked the whole activation chain in W, so W cost a full backward |
| greedy event loop, `defer_w`, `defer_w_cap`, `extra_inflight`, policies | a solved per-slot F/B/W order (`schedule_order`), executed in order with runnable-W fill on late inputs |
| interleaved 1F1B (`interleave`) | — (zero-bubble V schedule is the planned successor, see gaps) |
| ring with wrap (non-p2p) transport | the authenticated duplex chain (`p2p` no longer a flag) |
| dynamic claim pool (`stealing`, `replay_claims`, CLAIM/ROUTE frames) | static IWRR ownership weighted by profiled peer throughput (SWARM's routing rule) — a dynamic pool cannot be scheduled |
| 1F1B fill/drain microbatch formula | the bubble of the solved schedule |

Defects found while doing it:

| Defect | Fix |
|---|---|
| The deferred-dW store was thread-local; CUDA and MPS run backward on an autograd device thread, so dW was silently returned to a call that discarded it (found by the CPU + MPS fleet test) | process-wide store |
| With heterogeneous per-stage costs, the warm-up could give a downstream stage more F than its upstream had produced (the reference never sees it: it uses one median cost; there an all-zeros time table would hide it) | warm-up counts clamped to the upstream stage; 2,000 random heterogeneous configurations validated |
