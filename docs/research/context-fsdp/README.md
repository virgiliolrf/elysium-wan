# Context for the byte-exact FSDP / ZeRO-3 ADR (2026-10-09)

## What exists (main, approved design)
- Pipeline parallelism over a WAN (zero-bubble F/B/W), wide stages (several peers per stage, microbatches owned by
  throughput weight), ZeRO-1 inside a stage: peer k owns params[k::w] (whole tensors) with their fp32 master, m and v;
  each owner sums all M per-microbatch gradients of its shard in microbatch index order (OrderedAccumulator, fp32);
  parameters are broadcast after the update. Weights are replicated on every peer of a stage (no ZeRO-2/3, no TP).
- M2 phase 1 (merged): per-step telemetry (per-op times, per-hop wire time and bytes), the input a planner needs.
- Human rules: byte-identical to the reference (CLAUDE.md product rule; no reduction order chosen by topology, arrival
  order or worker speed; no lossy method, no DiLoCo, no async/stale updates); production precision bf16/fp8 inputs with
  fp32 master weights, optimizer state and outputs (fp32 compute allowed; nothing below); one decision per ADR, at most
  400 lines (an over-budget ADR is split); speed within a few percent of homogeneous pipeline parallelism.
- gpu false: theory and CPU simulation only (localhost multi-process clusters, emulated WAN); real hardware pending.

## The human's request (2026-10-09)
Make FSDP byte-wise exact, and let the planner place it:
- small batch or weak links: FSDP stays inside a site, and only pipeline boundaries cross the WAN;
- large pretraining (millions of tokens per site per step) with ~10 Gbps links: FSDP or data parallel with ZeRO may
  cross the WAN, still byte-exact;
- the planner decides with the FLOPs-per-byte arithmetic below, against each link's measured bandwidth (M2 telemetry).

## First-principles arithmetic from the discussion (DERIVED; verify and refine)
- FSDP/ZeRO-3 per peer per step: all-gather parameters twice (forward, backward) and reduce-scatter gradients:
  ~10 bytes x P_stage (bf16 gathers + fp32 gradients), against 6 x P_stage x T FLOPs (T = tokens that peer processes
  per step): ~0.6 x T FLOPs per byte, growing with the batch.
- Pipeline boundary: ~4h bytes per token (activation forward, gradient back), against ~72 h^2 L FLOPs per token
  (L = layers in the stage): ~18 h L FLOPs per byte, independent of the batch.
- Bandwidth to keep a 1 PFLOP/s device busy ~ device FLOP/s / (FLOPs per byte): FSDP at 256k tokens per peer per step
  ~52 Gbps, 1M ~13 Gbps, 4M ~3.3 Gbps; pipeline at h = 8k, L = 4 ~14 Gbps at any batch. Break-even T ~ 30 h L tokens.
- Limits: the critical batch size bounds T; RTT 50-200 ms is hidden by prefetching gathers layers ahead (one RTT per
  step remains).
- Exactness, term by term: the parameter all-gather is a byte copy (exact); forward and backward with full parameters
  equal the unsharded computation; the gradient reduce-scatter is the only arithmetic and must keep the recipe's
  canonical order (microbatch index order, fp32, mean division as the recipe defines), never NCCL ring/tree order,
  never reduced precision on the wire; the optimizer is elementwise per shard (exact). Stock PyTorch FSDP is
  deterministic for a fixed configuration but not byte-exact across topologies (ring order depends on world size,
  gradient pre/post-division by world size, optional bf16 reduction).
- Tensor parallelism contrast: column-parallel splits are exact; row-parallel splits the K reduction and changes bits
  unless its degree and combine order are fixed by the recipe.

## Related draft (not granted, from the paused heterogeneous track)
- draft-hetero-ADR-0002-fold-tree-element-range-zero1.md and draft-hetero-fold-tree-notes.md: a recipe fold tree and
  element-range ZeRO-1 that cut wide-stage exchange volume (379 -> 105 ms at 1B, w = 8, 100 Gbps, DERIVED). Its review
  was unstable and it conflicts with M2 phase 2's speed-weighted ownership; reuse ideas, do not adopt it wholesale.
- M2 phase 2 (next) re-plans microbatch ownership between steps from telemetry; the sharding decided here must compose
  with it (ownership changes must not change a reduction order).
