"""wantrain: synchronous, lossless pipeline training over wide-area networks
on heterogeneous hardware.

Subsystems:
  runtime/          stage runtime (zero-bubble F/B/W execution, recompute,
                    wide ZeRO-1 stages with weighted microbatch ownership),
                    JobSpec, fixed-order accumulation, determinism
  transport/        frames, safe object codec, WAN shim, peer identity,
                    coordinator control channel
  codec/            lossless boundary wire codecs (byte planes, scaled fp8)
  model/            flat-layer-list models, precision recipes, data,
                    deferred weight gradients
  scheduler/        zero-bubble schedule, selection, chain routing,
                    cost-weighted partitioning, microbatch solver,
                    precision/memory admission, job planner, placement
  coordinator/      control plane: registry, launch, monitoring, recovery
  worker/           node agent and the supervised stage process
  resources/        hardware and link discovery, F/B/W profiler, memory fit
  fault_tolerance/  liveness judging, checkpoint waves, replacement, audit
  validation/       reference trainer (oracle), bitwise comparator, model
                    contract checks
  launch/           local multi-process launcher
"""

__version__ = "0.1.0"
