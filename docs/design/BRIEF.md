# BRIEF: distributed-training engine over WAN

Request specification compiled from the human's instructions (2026-10-07 and 2026-10-08), quoted where they set intent.
Anything inferred is marked ASSUMPTION. The human owns this file: edit it freely; the design follows it.

# Goal
Train neural networks (decoders) efficiently across heterogeneous machines connected by wide-area networks. The
engine and the distributed scheduler only. In the human's words: "Build only the distributed-training engine and
distributed scheduler". The execution has a single path, zero-bubble pipeline parallelism: "unifique a arquitetura para
ter só um caminho tipo só zero-bubble pipeline paralelism e não 1f1b" ("unify the architecture to have a single
path, only zero-bubble pipeline parallelism and not 1F1B").

# Existing Architecture
See [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) and [ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md): the `wantrain`
package merged in `9f7d81c`.

# Problem
Datacenter interconnects are expensive and scarce. Idle GPUs exist in many places, of many kinds, behind slow and
variable links. Existing pipeline engines assume a fast, uniform fabric. On a WAN, their bubbles and synchronous waits
dominate.

# Non-goals
"Do not build the website. Do not build the product. Do not build the marketplace. Do not build billing. Do not build
the Elysium SaaS layer." Also excluded: lossy, asynchronous, local-step or order-changing methods (CLAUDE.md, product
rule).

# Proposed Change
Covered by DESIGN.md: synchronous lossless pipeline; zero-bubble F/B/W orders solved from measured per-stage costs and
per-hop communication; compute-weighted, memory-feasible cuts; throughput-weighted wide stages; heal from spares.

# Components Affected
runtime, model, scheduler, transport, control (DESIGN §7).

# Interfaces
CLI: `wantrain coordinator|worker|stage|local|plan|compare|validate-model|prepare-data`. Job: `JobSpec` JSON or a plan
job. Results: hex losses, checkpoints, telemetry.

# Data Flow
Activations travel forward and gradients backward, slot to slot, over authenticated TCP. Checkpoint waves go from the
agents to the coordinator.

# Control Flow
Coordinator: discovery → selection → routing → planning → profiling → schedule → launch → monitor → heal or restart →
complete.

# Failure Modes
"fault tolerance": a worker or stage dies, freezes, or disconnects; partitions; stragglers. ASSUMPTION: losing at most
`ckpt_every` steps of work per failure is acceptable. ASSUMPTION: coordinator loss stops the run until M4.

# Security Model
Workers may be rented machines on the public internet; links are public; the operator controls the coordinator.
"o cyber agent e segurança deve tentar ao máximo from first-principles aumentar segurança sem trade-offs como velocidade
significativos". Every mitigation is costed against the step time (THREAT_MODEL §3).

# Performance Requirements
"WAN-first", "heterogeneous hardware", "real distributed training". ASSUMPTION: the target is high GPU utilization at
cross-region latencies of 20–150 ms one-way, with per-hop bandwidth at or above the busy-GPU line of DESIGN §9.

# Compatibility Requirements
Python ≥ 3.10, `torch ≥ 2.4`; CUDA, MPS and CPU devices; mixed device maps declared in the spec.

# Testing Requirements
"validate node lifecycle, scheduler, worker failure, heterogeneity, real distributed training"; "preserve real
scheduler (no mocks)".

# Acceptance Criteria
1. `node agents-architetura/tools/gates/ladder.mjs` is green: unit, integration, chaos, full, invariants, static and
   security.
2. Every integration and chaos run's losses are bit-identical to the single-device oracle: `max abs diff = 0`.
3. The zero-bubble path is the only execution path: `grep -rn "one_f_one_b\|split_backward\|stealing" src/` is empty.
4. M1: on 2–4 GPUs in two regions, the measured step time is within 25% of the prediction, and tokens/s are reported
   against the same GPUs in one site. ASSUMPTION: 25% is the tolerance until the profiler is calibrated.
