# wantrain — distributed training engine over WAN

`wantrain` trains a decoder with **zero-bubble pipeline parallelism** across
heterogeneous machines connected by wide-area links. The model is cut into
contiguous layer slices, one slice per stage; stages exchange boundary
activations forward and activation gradients backward over TCP. The engine's
exactness contract: on identical hardware the per-microbatch loss sequence is
**bit-identical** to a single-device run of the same recipe, whatever the
partition, the schedule, the network, a crash or a heal.

It is the distributed-training engine extracted from Elysium
(`virgiliolrf/Efficient-Distributed-Training-with-Heterogeneous-Hardware-over-Wide-Area-Networks`);
see [MIGRATION.md](MIGRATION.md) for what was taken, adapted and left behind.

## One execution model: zero-bubble pipeline parallelism

There is exactly one way a step executes (Qi et al., *Zero Bubble Pipeline
Parallelism*, ICLR 2024):

- **F**: the stage forward (under `no_grad` when `recompute` keeps only the
  boundary).
- **B**: the input gradient only. Each linear layer computes `dX = dY·W` and
  leaves `dW` pending (`model/wgrad.py`, the reference implementation's
  WeightGradStore); the input gradient goes upstream at once; the graph is
  freed. B is the critical path.
- **W**: the pending `dW = dYᵀX` GEMMs of one microbatch, folded into the
  step gradient in microbatch order. W has no cross-stage dependency, so the
  schedule places it where 1F1B would leave a bubble.

The order of F/B/W for every slot is **solved**, not improvised
(`scheduler/zero_bubble.py`): a port of the paper's automatic scheduling
heuristic (sail-sg `zb.py`: warm up with as many F as memory allows until the
first B can arrive, alternate F/B, insert W where it fits before the next
input, keep each stage at least one F ahead of the next, grid search over three
choices), generalized to **per-stage costs** and **per-hop communication** for
heterogeneous hardware over a WAN. On the reference's own inputs (uniform
costs) the port is identical to the original on 2,400 randomized
configurations. The runtime executes the order; when an input is late it pulls
forward an already-runnable W (never violating a dependency), which absorbs WAN
jitter and cost-estimate error.

There is no 1F1B path, no greedy event loop, no monolithic or two-call
backward, no interleaving and no dynamic claim pool: every spec carries a
schedule (`JobSpec.schedule_order`), and `finalize()` solves one from the
analytic cost model when a spec arrives without it.

## Architecture

A control-plane / data-plane split. One coordinator plans and supervises and
never touches numerics; training traffic flows worker to worker along the
chain. Inside a *wide stage* (several peers holding one slice) microbatches are
owned by peers in proportion to their measured throughput.

```
                         ┌──────────────────────────────────────────┐
  job (recipe or plan) ─▶│ Coordinator (control plane, no numerics) │
                         │ registry · selection · link probing ·    │
                         │ chain routing · planner · F/B/W profiling│
                         │ · zero-bubble solve · launch · liveness ·│
                         │ checkpoint store · heal / restart        │
                         └───┬───────────┬───────────┬──────────┬───┘
            control channel  │ HELLO/HB  │ ASSIGN    │ CKPT     │ (spares idle,
            (JSON, token)    ▼           ▼           ▼          ▼  registered)
                       ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
                       │ agent A │ │ agent B │ │ agent C │ │ agent D │
                       │ stage 0 │ │ stage 1 │ │ stage 1 │ │ spare   │
                       │ (cuda)  │ │ peer 0  │ │ peer 1  │ │         │
                       └────┬────┘ └──┬───┬──┘ └──┬───┬──┘ └─────────┘
   data plane (TCP,         │  F  ▶   │   └ZeRO-1─┘   │  ▶ stage 2 ...
   WAN, peer-authenticated) │ ◀ B     │ IWRR-weighted │
                            └─────────┴── microbatch ownership; each slot
                                          runs its own F/B/W order
```

## Package map

| Package | Responsibility |
|---|---|
| `runtime/` | `JobSpec` (the single JSON object that determines a run), `StageRuntime` (executes a slot's zero-bubble order; wide ZeRO-1 stages; adaptive microbatching; checkpoint/resume), `OrderedAccumulator` (fixed-order folding, audit quarantine), determinism |
| `model/` | flat layer-list decoders (built-in GPT, Llama-family, HF checkpoints, user `build(cfg)`), LoRA, precision recipes, deterministic data, `wgrad` (deferred weight gradients) |
| `scheduler/` | `zero_bubble` (heuristic, simulators, IWRR ownership, engine bridge), node selection, link memory, chain routing, cost-weighted memory-feasible partition DP, microbatch solver, precision/memory admission, planner, placement |
| `transport/` | frames + fragments + END handshake, safe object codec, WAN shim, ed25519 peer identity with allowlist, coordinator control channel |
| `codec/` | lossless boundary codecs (byte planes, scaled fp8) |
| `resources/` | capability probe (CUDA/MPS/CPU), F/B/W profiler, link probes + duplex + WAN emulation, memory fit check |
| `fault_tolerance/` | liveness judge, checkpoint wave store, replacement choice, audit policy |
| `validation/` | reference trainer (the oracle), bitwise comparator, model contract checks |
| `launch/` | local multi-process launcher |

## Training-job lifecycle

```
job ─▶ discovery: agents register measured capabilities
    ─▶ selection: one region, distinct addresses, capability window, link
       memory first, fastest first; the rest are spares
    ─▶ routing: probe matrix → chain maximizing its worst hop → full
       bidirectional measurement per hop (duplex) → re-route on disagreement
    ─▶ planning: recipe mode places a JobSpec (layers ∝ speed, per-layer
       compute costs, memory-feasible cuts); plan mode solves precision,
       microbatch size and microbatches per step (from the bubble the
       zero-bubble schedule leaves on the measured link) and measures the
       peak memory on each node
    ─▶ profiling: every slot times its F, B and W on its own device; the cut
       is refined by measured stage throughput (if it helps > 5%); wide
       stages get IWRR ownership weighted by peer throughput
    ─▶ schedule: zero-bubble orders solved for those costs, the hop
       communication and each stage's activation budget; pinned in the spec
    ─▶ launch: one spec to every slot, last rank first
    ─▶ per step: each slot runs its F/B/W order; B's input gradient crosses
       the WAN at once, W fills the bubbles; gradients fold in microbatch
       order; AdamW (ZeRO-1 in wide stages)
    ─▶ checkpoint every ckpt_every steps (shard + optimizer + loss history),
       shipped and sha256-verified to the coordinator
    ─▶ completion: losses (hex-exact) and telemetry collected
```

## Scheduler

1. **Resource discovery**: matmul TFLOPs, memory, compute capability per node;
   region and uplink declared.
2. **Selection** (`selection.py`): dead-host memory, device types, capability
   window, memory and uplink floors; group by region and hardware; rank by
   measured path verdicts, bf16, total speed.
3. **Routing** (`partition.py`): `order_stages`, exact with branch & bound to 8
   nodes, greedy beyond; measure-and-reroute in the coordinator.
4. **Partition**: exact min-max DP over contiguous ranges with per-layer
   compute costs (the head's logits GEMMs at a 32k vocabulary cost ~3 blocks)
   and a memory-feasibility predicate per stage.
5. **Ownership** (wide stages): interleaved weighted round robin — SWARM's
   routing rule, made static so the schedule can plan it.
6. **Zero-bubble schedule** (`zero_bubble.py`): stage-level orders from
   aggregate stage costs, split per peer by owner (each peer keeps the stage
   order's relative sequence, so the split cannot deadlock); validated by a
   slot-level simulator.
7. **Microbatches per step** (`schedule.py`): the smallest M whose solved
   schedule idles at most the bubble target (e.g. 4 stages need 7 for ≤ 5%,
   where the 1F1B fill/drain would need 57).
8. **Admission** (`autoconfig.py`): every boundary precision rung priced
   against the measured link (overlap condition) and memory; lossless rungs
   preferred, refusals name the number that refused.

## WAN-specific mechanisms

| Mechanism | Where |
|---|---|
| Zero-bubble schedule with per-hop communication in the solve | `scheduler/zero_bubble.py` |
| Input gradient sent before weight gradients; W fills bubbles | `runtime/stage.py`, `model/wgrad.py` |
| Late-input W fill at runtime (jitter) | `runtime/stage.py::_fill_w` |
| Bandwidth floor / precision ladder admission | `scheduler/autoconfig.py`, `planner.py` |
| Activation recomputation from the boundary | `runtime/stage.py` |
| Lossless entropy-coded boundary wire (zstd, byte planes, scaled fp8) | `codec/` |
| Adaptive microbatching (fragment size halves when a send escapes budget) | `runtime/stage.py::_sender_loop` |
| One duplex authenticated hop, single dial direction (asymmetric NAT) | `runtime/stage.py::_connect_chain` |
| Parallel TCP streams per hop with exact reassembly | `runtime/stage.py::_wire_multi` |
| Probe matrix, bidirectional hop measurement, duplex classification | `resources/links.py` |
| Chain routing by worst hop, re-route on disagreement | `partition.py`, coordinator |
| Link memory and dead-host memory across runs | `scheduler/linkmemory.py` |
| Per-link WAN emulation (latency, bandwidth, jitter) | `transport/shim.py`, `resources/links.py` |
| Reconnect grace for control-plane blips | agent + liveness |

## Heterogeneous hardware

Layers in proportion to measured speed and per-layer cost; a memory predicate
inside the partition DP; per-slot device map (CUDA, MPS and CPU stages in one
run); per-stage F/B/W costs from profiling feed the schedule, so a slow stage
gets the warm-up and W placement its own costs call for; wide stages own
microbatches in proportion to peer throughput; the planner works on the
weakest node's compute capability and the tightest memory; replacement prefers
the identical model. Mixed backends train losslessly but are bit-comparable
only to a run on the same device map.

## Fault tolerance

- **Detection** (`fault_tolerance/liveness.py`): control connection lost past
  the reconnect grace; stage killed without its done marker; a slot that lost
  its peer while another is silent; heartbeat silence (two strikes); fleet-wide
  stall. Engine exceptions fail the run; simultaneous failures are refused as a
  network problem.
- **Heal**: abort survivors (each ships its last wave), pick a spare, take the
  newest wave every slot holds, restore it (sha256-verified), relaunch with
  `resume=true` under the original spec — schedule, ownership, partition and
  seeds pinned. Budget: `max_heals`.
- **Restart in place**: everyone alive but no progress → same resume on the
  same nodes. Budget: `max_restarts`.

## Running

```bash
uv sync
WANTRAIN_COORD_TOKEN=secret uv run wantrain coordinator --port 28000 --job job.json
WANTRAIN_COORD_TOKEN=secret uv run wantrain worker --coordinator COORD:28000 \
    --advertise-host PUBLIC_IP --data-port 29000 [--region eu-west]
uv run wantrain local --spec spec.json --compare      # every slot on this machine, diffed
```

## Validation

`tests/unit` (scheduler vs the paper's formulas and the reference
implementation's own schedules, heterogeneous stress, IWRR, deferred-dW
bit-exactness incl. device threads, planner, liveness, checkpoints),
`tests/integration` (every mechanism as a real multi-process run diffed
bit-for-bit against the oracle; coordinator end-to-end incl. WAN-emulated
routing, profiled schedules, wide stages and a real CPU + MPS fleet),
`tests/chaos` (worker death, stage kill, frozen stage, control blip, no spare,
wide-stage peer death — healed runs stay bit-identical).
