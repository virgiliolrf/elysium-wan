# SPEC-scheduler history

- 2026-10-09 — ADR-0006 (exact sharded stages; awaiting the human grant). New invariants SCHEDULER-I16 (admission
  by device class, the stream test S1, the exposure test S2 and memory) and SCHEDULER-I17 (the simulator charges `X_s`),
  both DEFERRED-TO: M2b. Task 4 rewritten from "M2 — plan mode for wide stages" to the M2b sharded plan mode.
- 2026-10-09 — ADR-0006 round 1 (review findings; awaiting the human grant). SCHEDULER-I16: S1 per link (NIC and site
  uplink, capacity under concurrent load), S2 from the simulation, device and host memory as separate conditions.
  SCHEDULER-I17: per-unit exchange ops on FIFO links instead of one `X_s` block. Task 4: `shard: "pieces"`, choice by
  simulated step time.
- 2026-10-09 — ADR-0006 round 2 (review findings; awaiting the human grant). SCHEDULER-I16: device class is the same
  model, also at re-plan; S1 sums every stage on a shared link and includes PCIe paths; device memory charges in-flight
  boundaries, the pending W and the head's logits against `vram_reserve`; host memory is the one-step bound (no
  measured δ), refusal `MEM-HOST`. SCHEDULER-I17: per-owner queues with need-first priority, per-peer op times,
  emission order chosen by the simulation. Task 4: the deepest cut is a candidate; equal-width wide stages get
  lane-consistent orders.
- 2026-10-09 — ADR-0006 round 3 (review findings; mechanism frozen; awaiting the human grant). SCHEDULER-I16: host DRAM
  is a link with the transport's counted passes per byte, summed over the GPUs of a host; device memory adds the
  own-piece hold `b_g·M_k·P_s/w`; host bound per peer under weighted IWRR; a re-plan that changes `M_k` re-checks
  memory. SCHEDULER-I17: the exchange is simulated as built (per-destination need-first send queues, links shared
  max-min fairly among chunks in flight), with the owner's fold and AdamW as op times before a gather's release.
- 2026-10-09 — ADR-0006 round 4 (review findings; mechanism frozen; awaiting the human grant). SCHEDULER-I16: S1 is
  `Σ k_ℓ·(b_g·P_s·share + A) ≤ cap_ℓ·c` (boundary bytes take the same k passes on host DRAM), with the host-DRAM share
  `2·(w−1)/w` written out; S2's C is the pipeline's compute per step at its slowest stage. SCHEDULER-I17: the Embed's
  tables are one fp32 unit released when W(mb) starts. Task 4: the candidate set keeps today's `zero1` wide stages and
  fewer-device plans, priced by the same simulation, so a job planned today keeps a plan; the lane-consistent-order
  bullet is removed (the engine-bridge defect stays a house finding, a prerequisite: ADR-0006 R14).
- 2026-10-09 — ADR-0006 round 5 (review findings; mechanism and predicate frozen; awaiting the human grant).
  SCHEDULER-I16: S1's window is the stage's own `c_s = 8·P^F_s·T_mb/F` (GEMM parameters), the definition S2 and the
  ADR's model already used; the S1 bullet had written `c = 8·P_s·T_mb/F` (all parameters), 1.73× too long at point B's
  stage 0, which would have admitted plans the ADR's model refuses. With several stages on one link S1 is written as a
  sum of rates, each stage at its own `c_s`, over the GPUs actually placed on the link; `b_g·P_s` is the per-unit sum.
  Host memory (MEM-HOST) is summed over every peer placed on a host and gains one step of gathers out and in
  (RUNTIME-I13), plus pinned chunks in flight; it was charged per peer alone. Derivation: ADR-0006 `round5.md` §1, §4.
- 2026-10-09 — M2b handout SC-2 (HANDOUT-M2b.md; awaiting the human grant with the plan). New SCHEDULER-I19
  (DEFERRED-TO: M2b): lane-consistent orders per peer of a wide stage (ADR-0006 R14, M2b's first task by the human's
  decision). Task 4 named "its own invariant" without an id; claiming SCHEDULER-I5 would have been a silent change.
- 2026-10-09 — M2b, the human's ruling at the plan grant (ADR-0006 R16; HANDOUT-M2b). SCHEDULER-I16: `c_s` in S1, S2
  and C is the planner's own per-microbatch cost (`layer_costs` through `zero_bubble.analytic_costs`, attention
  included, recompute as the recipe runs), one value everywhere. Was `8·P^F_s·T_mb/F` in S1 and S2, which with R16's
  attention-inclusive planner would have put S1 and S2/C on different windows. Point B stage 0 at k 10 moves 944.8 →
  809.8 GB/s per GPU (`docs/experiments/m2b-e7/PLAN.md`); no E7 choice changes.
- 2026-10-09 — ADR-0007 (slice-only build; awaiting the human grant). New SCHEDULER-I18 (DEFERRED-TO: M4): at a first
  launch the selection shares one init class (`INIT-CLASS` with per-class counts) and the build's host memory is charged
  per host (`MEM-INIT`), `4·(P_layer,max + P_tied)` per slot for a slice build and `4·P` for a full build.
- 2026-10-09 — ADR-0007 round 1 (review findings; awaiting the human grant). SCHEDULER-I18: the init class is the
  record's sampler fingerprint, used as a selection filter like `same_region` (refusal only when the record's group
  cannot field the plan, with per-fingerprint counts), never taken from the selection; `MEM-INIT` charges
  `4·P_layer,max` per slice slot.
- 2026-10-09 — ADR-0007 round 2 (review findings; awaiting the human grant). SCHEDULER-I18: selection applies CONTROL-I19's single
  exclusion rule (fingerprint from `HELLO`, and nodes a slot report excluded); `MEM-INIT` charges
  `4·(P_layer,max + P_held)` per slice slot.
- 2026-10-09 — ADR-0007 round 3 (review findings; awaiting the human grant; no mechanism changed). SCHEDULER-I18:
  `MEM-INIT` charges per slot the larger of the build charge and the resume charge (the slot's wave as `_resume`
  loads it, `12·P_s` for a whole-tensor AdamW slot), each with `4·P_s` for a CPU-device slot, and names the binding
  term; "a resume charges neither condition" is replaced, because with the slice build the resume load, not the
  build, binds host memory (ADR-0007 R7).
- 2026-10-09 — ADR-0007 round 4 (review findings; awaiting the human grant; no mechanism added). SCHEDULER-I18: "a
  resume compares no fingerprint" is replaced: a resume excludes no node from init, but a CPU-device slot's replacement
  in a run with a record must have the record's fingerprint (CONTROL-I19), because that slot computes with its CPU class
  (ADR-0007 R9).
- 2026-10-09 — merge of main (50d34cd) into factory/m2b. ADR-0007 took SCHEDULER-I18 on main (slice-build admission,
  DEFERRED-TO: M4); M2b's lane-order invariant (SC-2, ADR-0006 R14) is renumbered SCHEDULER-I19, text unchanged, and
  so are its references in task 4, PLAN-M2b, HANDOUT-M2b and the `covers:` lines of its tests.
- 2026-10-09 — conflict-degrade-memory (M2, task degrade-memory). SCHEDULER-I13 and task 1: "the largest feasible
  warm-up" was wrong. Non-increasing warm-ups that are all at their largest are equal under a uniform budget, and equal
  warm-ups put a hop's round trip on every microbatch: the 7B example at 8 GB replays `[316]×8` to a 98.28% bubble,
  against 17.31% for the gap search (measured by the researcher with `_seven_b` on the tree under review). §2.1 gains
  the warm-up family (gaps `g`, `k_j = min(L_j, 1 + g(p−1−j))`) and the selection rule (best of grid and family at
  the given budget). I13 now states what is guaranteed (fit, validity, reported bubble, refusal only below one
  microbatch) and what is not: a non-increasing bubble as the budget grows. The paper's grid and the family are built
  from the budget, and the researcher reproduced counterexamples (4×32: 16.52% at 5.95, 17.95% at 6.1; 3×12 with `mem_b`
  −0.9: 12.20% at 4.9, 14.29% at 5.05). A search that guarantees it (a fixed budget-independent candidate set, or the
  paper's ILP) was rejected for M2: a candidate set fixed across heterogeneous per-stage budgets loses the per-budget
  heuristic, and an ILP adds a solver for a 1–2 point bubble effect on timing only (bytes are unaffected). Reverse if a
  plan_job placement or width choice is shown to flip because of this non-monotonicity.
- 2026-10-09 — Conflict from replan-solver (M2), amended. SCHEDULER-I15 now defines the measured hop term as seconds
  per transfer, `busy_ms / n̂`, with `n̂ = frames` at `p2p_streams == 1` and `n̂ = bytes / _boundary_bytes(spec)`
  otherwise. Why: HANDOUT-M2 (Rebalancing) and PLAN-M2 replan-solver wrote "seconds per transfer (busy_ms / frames)",
  which holds only on the single-stream path (stage.py `_sender_loop`: one `send` event per transfer). On the
  multi-stream path `_frag_sender` logs one `send` per 1 MiB fragment and telemetry.py `summarize_step` counts
  `frames += 1` per event, so `busy_ms / frames` is seconds per fragment, an underestimate by ⌈payload/1 MiB⌉ (8× for
  an 8 MiB boundary). The PLAN's governing quantity ("seconds per transfer") wins over its parenthetical. Rejected: a
  `transfers` count in telemetry.py (exact for every codec, but no M2 task owns telemetry.py, so it needs a new
  handout, for a cost estimate whose output never touches the training bytes); recovering the count from
  `frames` and `bytes` alone (not determined when lossless codecs vary the payload). Trade-off: on multi-stream, `n̂`
  inherits the coded-bytes model's error (exact for `raw` when the boundary tensor is the boundary dtype; the
  codec-ratio error for `planes`/`fp8_planes`), the same error the modelled term already carries; both predictions of
  the 10% rule use the same costs. Reverse if a test shows `n̂` off by more than 10% from the true transfer count on
  a pinned recipe, then add the telemetry field through a handout. The implementer's latency reading is confirmed:
  `hop_comm_seconds` adds the one-way latency to a measured hop.
- 2026-10-09 — HANDOUT-M2 §8 (P1) recorded on top of ADR-0007; M2 ids CONTROL-I19->CONTROL-I23, RUNTIME-I14->RUNTIME-I17 (collision with ADR-0007) (no scheduler id is renumbered; A10's cross-reference to the rule names SPEC-control task 2 and
  CONTROL-I21). Applied: **A10** — SCHEDULER-I13 and SCHEDULER-I15 lose `(DEFERRED-TO: M2)` (both already covered by
  tests in `tests/unit/test_zero_bubble.py`); task 1 appends "`solve()` uses the measured budget as given (no 1F1B
  floor); refuse only below one microbatch, naming the bytes."; task 2 appends "`replan(spec, windows,
  min_gain=0.10)`; the threshold is A7's rule", with "(HANDOUT-M2 §8 A7: SPEC-control task 2 and CONTROL-I21)" added so
  the reference resolves inside the specs. Already present from earlier task conflicts: the SCHEDULER-I13 selection
  rule and the SCHEDULER-I15 per-transfer hop term (kept unchanged).
- 2026-10-09 — ADR-0008 round 0 (awaiting the human grant). §2.1: MoE pending W. §2.3: MoE block cost `6·P^F + 12·h·s`.
  SCHEDULER-I21 (new): compute at active, bytes and memory at total parameters; expert units grouped by projection in
  SCHEDULER-I17; ρ_s named on refusal. Task 5 (M2c).
- 2026-10-10 — ADR-0008 round 1 (awaiting the human grant). M2c → M4 in §2.1, §2.3, SCHEDULER-I21 and task 5; task 5's
  stale reference "as I19 states" corrected to I21 (SCHEDULER-I19 is M2b's).
- 2026-10-10 — ADR-0008 round 2 (awaiting the human grant). §2.3: an MoE block's analytic layer cost omits per-expert
  launches and host count reads (a lower bound); the solve uses profiled costs and the re-plan corrects the cut.
  SCHEDULER-I21: the count lane's bytes and the biases in SCHEDULER-I17's simulation; `c_s` stays the analytic
  count, so S1 and S2 are conservative for MoE stages; a plan records each MoE stage's analytic `c_s` next to its
  profiled time.
- 2026-10-10 — ADR-0009 round 0 (awaiting the human grant). §2.3: the HF boundary dtype is HF's residual dtype, not
  declared; a paragraph on HF models and tied tables (costs from the skeleton, eager attention memory, the Head stage's
  table and accumulators, the tied link's bytes and exposed time). New SCHEDULER-I22. Task 6.
- 2026-10-10 — ADR-0008 round 3 (awaiting the human grant; mechanism frozen). SCHEDULER-I21: under HF modules the two
  3-D expert units are produced in B, released when W(mb) starts at the master dtype's bytes (4 under today's
  recipe), and their dW is charged to B, not W.
- 2026-10-11 — ADR-0009 round 1 (awaiting the human grant). §2.3 precision: a declared `boundary_dtype` other than
  HF's residual dtype is refused at validation. HF models and tied tables: the Head stage's and stage 0's tied memory
  (held rows bounded by `n_pend·min(V, n_pend·T_mb)·h·4`), the forward- and backward-direction bytes on every chain hop
  between them, and `T_tied` on the step's critical path. SCHEDULER-I22: no link outside the chain planned or probed;
  the rationale names the tied hops and `T_tied`. Task 6 rewritten to match.
- 2026-10-11 — ADR-0008 round 5 (awaiting the human grant; mechanism frozen). §2.1 memory model and SCHEDULER-I21:
  under HF modules each MoE block's bank gradient (`b_m·E·3hf` bytes) is charged in `mem_b` and freed in `mem_w`, and
  SCHEDULER-I16's device-memory condition adds `m_BW,k·b_m·P_bank,s` (`M_k` before an order is solved); before this
  the planner charged none of it (5,376 MiB per Mixtral layer per outstanding microbatch).
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant). §2 HF models: per-unit window costs as PROFILE measures
  them; eager attention `b·H·S²·s` with `s` = 6 under autocast, 4 in fp32, plus the softcap term (was `·2`); tied bytes
  per hop `|T_i|·h·4` forward and `|U_{t+1}|·h·4` backward; `ΔT_tied` with throughput, warmup, round-trip and +0 scan
  terms (was `T_tied`, the round trip only). SCHEDULER-I22 and task 6 renamed to `ΔT_tied` and `s`.
- 2026-10-11 — ADR-0009 round 3 (awaiting the human grant; mechanism frozen). SCHEDULER-I22: a plan with a tied table
  across stages and stage 0 or the Head stage at width > 1 is refused, naming it, until RUNTIME-I19 specifies `TIED`
  routing to the fold owners (ADR-0009 R14, open).
- 2026-10-11 — ADR-0009 round 4 (awaiting the human grant; mechanism frozen, text corrected). §2 `ΔT_tied` gains its
  fifth term, the backward link's row burst `R_1..R_{M−1}` past the link's idle window `W_idle` (round 3 omitted it;
  0 at Gemma2-2B S 4, M 8, L 20 ms, 283 of 751 ms at M 16, L 0), and `t_s` includes the tied head GEMM on the Head
  stage (round 2 used the non-embedding parameter count). SCHEDULER-I22's rationale names five terms.
- 2026-10-10 — SC-5 (HANDOUT-M2b precondition, the researcher's). SCHEDULER-I16's host-DRAM k amended from "10 today"
  to "15 today on the sealed M2 data plane". COUNTED (no measurement) on `factory/m2b` after the M2 merge (a62e4eb), by
  ADR-0006 `round3.md` §2's method: a user-space or kernel copy is 2 passes, a DMA 1. Exchange frame (OBJ): sender 12
  (D2H 1, `serial.py:66` 2, `serial.py:143` 2, `framing.py:117` 2, `encrypt_into` `secure.py:197/369` 2, kernel 2,
  NIC 1; the gather `sendmsg` at `secure.py:394` adds none), receiver 18 (NIC 1, kernel `recv_into` 2,
  `decrypt_into` `secure.py:207/572` 2, `SecureSocket.recv` `secure.py:633` 2, `framing.py:157-158` 4,
  `serial.py:162` 2, `serial.py:79` 2, `exchange.py:287` 2 for a tensor spanning 16 MiB chunks, H2D 1). Boundary frame
  (raw): sender 10, receiver 16; 12 and 18 when fragmented (`p2p_streams` > 1). S1 prices `k·2·per_dir`
  (`admission.py:322`), so k is the mean of the sides: 15 on the worst default path of either kind (14 for an exchange
  unit within one chunk, 13 for an unfragmented boundary; 17 for exchange frames under `adapt_budget_ms` > 0, off by
  default). Sealing added 2 on the sender and 4 on the receiver, not HANDOUT-M2 R4's 2 per side. Consequences
  (`sc5-passes.md`): at 15 a 1-GPU 600 GB/s host admits no GPU of point B's stage 0 or interior (bound k ≤ 13.4) and no
  stage-0 GPU of E7's (16, 2) plan (k ≤ 13.5), so DESIGN §9's "S 16, w 2 with pieces on 1-GPU hosts at today's 10"
  needs an amendment; the toy CPU fleet of `test_pieces_plan_oracle` passes S1 at 15 while the host's DRAM figure is
  above 30.0 GB/s at 1.6 TFLOPS (load/cap 0.65 at the MEASURED 46 GB/s). `admission.K_HOST_DRAM_COUNTED` must become 15
  (test `k_counted` reads this clause). E7/E4 scenarios keep their explicit k.
- 2026-10-10 — SC-5 follow-up (the orchestrator, under autonomy). SCHEDULER-I16's counted k is 17 when adaptive
  microbatching fragments the exchange frames (`adapt_budget_ms` > 0: SC-5 counted sender 14, receiver 20), 15
  otherwise; `admission.counted_k` prices it, and "all features on" (adapt_budget_ms 8) is admitted at 17.
- 2026-10-11 — ADR-0008 round 7 (correction pass, autonomy). §2.1 memory model: the HF v5 bank gradient is charged in
  `mem_b` in both memory branches and released in `mem_w`, so the replay holds `m_BW·b_m·Σ E·3hf` with `m_BW`
  microbatches between B and W (stated with the number: 22.5 GB per in-flight microbatch on a 4-layer Mixtral stage).
  SCHEDULER-I21: the hold is whole on every peer of a wide stage; the replay and I16's condition carry `m_BW`; a plan
  over budget with it is refused. ADR-0009 round 6: task 6 names its rollback target (tied HF at S > 1 refused at
  planning, with runtime 7a/7b; the boundary-bytes and eager terms revert with SPEC-model 4b).
- 2026-10-11 — M4 implement deviation (scheduler-m4). §2.3: HANDOUT I-13 froze `delta_t_tied` with 8 positional
  arguments, but the formula of record (`round5.md` §8, this §2.3) also needs h, V, T_mb and BW_mem. As implemented
  (`scheduler/autoconfig.py:684-744`; `TIED_ADAMW_BYTES = 28` at `:401`, 16 B read + 12 B written per element from
  `docs/research/adr-0009/scripts/tied_terms.py`): the 8 positional arguments unchanged, plus keyword-only `d_model`,
  `vocab_size`, `tokens_mb`, `mem_bw_Bps` and `adamw_bytes_per_elem=28`; `mem_bw_Bps = inf` (unmeasured) makes the
  scan and AdamW terms 0, a lower bound. HANDOUT I-13 amended alike.
- 2026-10-11 — the human's decision (~10:35, confirmed ~19:35): homogeneous fleets only (CONTROL-I25). §2.3 Selection
  step 1: a candidate fleet holds one device class, else `MIXED-FLEET` with per-class counts; `plan_job` and `place`
  refuse a fleet of more than one class before any cost is computed.
