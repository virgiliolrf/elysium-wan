# DESIGN history

- 2026-10-09 — ADR-0006 (exact sharded stages; awaiting the human grant).
  - §0: a "Planned (ADR-0006, M2b)" bullet.
  - §3: a "Sharded stages" row.
  - §8: milestone M2b.
  - §9: "Sharded stages" (the S1 and S2 tests, with numbers).
  - §10: a decision row.

  Editing DESIGN.md voids any grant of its previous text.
- 2026-10-09 — ADR-0006 round 1 (review findings; awaiting the human grant). Same five places, new text:
  - §0: per-tensor dim-0 pieces, streamed per unit, per-link and per-unit admission.
  - §3: working-copy dtype from the recipe; fold from an fp32 copy of microbatch 0's widened piece; quantum boundaries.
  - §8: M2b in two steps (`wire: native`, then `shard: pieces`); credits removed.
  - §9: topological links, per-link S1, per-unit S2 with one exposure expression, memory charged to device and host,
    cross-site admission domain (replaces the round-0 blanket refusal).
  - §10: decision row priced at the worked point.
- 2026-10-09 — ADR-0006 round 2 (review findings; awaiting the human grant). Same five places, new text:
  - §0: sharded stages serve forced width and memory; asynchronous copies; per-unit updates.
  - §3: emission order changes no bit; per-unit updates change when, not what; same device model after a heal.
  - §8: M2b adds asynchronous copies, the barrier removal, the one-step host bound, lane-consistent orders,
    class-preserving heal and shared pacers.
  - §9: depth against width per lane; S1 summed over stages on a shared link and over PCIe; per-owner simulation;
    cross-site admission domain withdrawn (no useful 7B placement).
  - §10: the decision row is priced against the deepest cut and at the forced-width point.
- 2026-10-09 — ADR-0006 round 3 (review findings; mechanism frozen; awaiting the human grant). Accounting corrected:
  - §0: "more devices than block cut points".
  - §8: M2b names need-first send queues, per-peer own-piece and host holds, `pieces` requiring `wire: native`, the
    logical-to-`zero1` reader and host DRAM in S1.
  - §9: depth against width with the fold every placement pays (55.5% vs 54.6%); sub-block cuts named as an unbuilt
    tie; host DRAM as a link (1.6 GPUs per 600 GB/s host today); S2 numbers as built (18.8 and 7.8 ms, step 1 alone
    24.6 ms); memory per peer with the own-piece hold and weighted IWRR; 1T fits only at bounded M/w.
  - §10: the decision row's numbers and the sub-block alternative.
- 2026-10-09 — ADR-0006 round 4 (review findings; mechanism frozen; awaiting the human grant). Accounting corrected
  (`docs/research/adr-0006/round4.md`):
  - §0: sharded stages serve forced width and width that shortens the step; the memory case is withdrawn.
  - §8: the M2 row no longer lists "plan mode for wide stages" (moved to M2b by ADR-0006 in round 0); the M2b row
    names boundary bytes in host-DRAM S1, `zero1` and narrower candidates, and the lane-order house fix as a
    prerequisite instead of an M2b deliverable.
  - §9: Embed and Head priced (the Head stage sets C, stage 0 binds); depth against width restated (S 16, w 2 at
    400 Gbps, depth at 100 Gbps); host DRAM 1.3 and 6.5 GPUs per host; S1 with boundary bytes at k; S2's C at the
    slowest stage; point-B numbers (2163.3 → 1124.5 ms); the planner's candidates when pieces are refused; sub-block
    cuts no longer a tie; the 1T fit claim removed.
  - §10: the decision row's numbers and alternatives.
- 2026-10-09 — ADR-0006 round 5 (review findings; mechanism frozen; awaiting the human grant):
  - §0: "where width shortens the step" qualified by host topology and host passes per byte.
  - §8: the M2b row's host holds are summed per host.
  - §9: `c_s = 8·P^F_s·T_mb/F` (GEMM parameters) replaces `c = 8·P_s·T_mb/F`; S1 written per stage at its own `c_s`
    and summed over the GPUs placed on each host; host DRAM restated per stage (stage 0: 0.64 GPUs per 600 GB/s host
    at k 10, 3.18 at k 2; interior 1.34 and 6.70), so point B's 1124.5 ms needs ≤ 6 passes and today's transport gets
    1721.3; 32-H100 width qualified by host topology; the host bound gains gathers (21.7 GB at A; 49.2 on B's stage 0)
    and is summed per host; own hold 32 B/param for the fp32 tables; the planner's fallback per host configuration.
  - §10: the decision row's conditions (host passes, host topology).
- 2026-10-09 — ADR-0007 (slice-only build; awaiting the human grant).
  - §3: a "Slice-only build" row (replay of the recipe's init stream, draw-free resume, one init class).
  - §8: the M4 row names ADR-0007 and its invariants (MODEL-I6, RUNTIME-I14, SCHEDULER-I18, CONTROL-I19).
  - §10: a decision row; open question 5 answered (no change of the init stream) and replaced by the init-portability
    question.
- 2026-10-09 — ADR-0007 round 1 (review findings; awaiting the human grant):
  - §3: the "Slice-only build" row now states the init record (ledger, per-tensor digests, sampler fingerprint), the
    replay of the writers only, the per-phase ledger check and the digest check; the declared class key is gone.
  - §8: the M4 row names ledger replay, the init record and init on the record's class.
  - §10: the decision row's numbers (88 s per 70B launch after a 676 s record; 84 slices and 168 ledger builds MEASURED)
    and alternatives on one table (do nothing at its strongest, recipe v2, seed checkpoint, a declared class key);
    question 5 restated for the record's fingerprint.
- 2026-10-09 — ADR-0007 round 2 (review findings; awaiting the human grant):
  - §3: the record is bound to (config, seed, init program) and its ledger anchored at `manual_seed(seed)`; its digests
    are taken by the phase-major walk at each tensor's last write; a slot checks each tensor once it is final.
  - §10: the record takes 708 s on one core (was 676 s with a second pass); alternatives with the record as a column
    (full-walk replay 603 s, recipe v2 on torch's samplers 88 s with a 620 s cheaper record) and a portable
    counter-based sampler costed (41–1739 s per launch by its per-sample cost, HYPOTHESIS).
- 2026-10-09 — ADR-0007 round 3 (review findings; awaiting the human grant; no mechanism changed):
  - §10: the host-memory claim restated for both homes and with the resume load (`12·P_s` per slot), which now bounds
    the host: 70B at S 32 on 80 GB devices needs 207 GB per 8-slot host instead of 2207 GB (device split HYPOTHESIS);
    the 70B S 8 "27.4 GB, a model larger than one host can start" claim is withdrawn (that plan trains nowhere, and
    with the home on the host its build alone needs 303 GB). Times completed with the record walk's memory passes:
    88 → 94 s per launch, record 708 → 811 s; alternatives 879 / 608 / 94 s, v2's record 710 s cheaper, a portable
    sampler 46–1745 s.
- 2026-10-09 — ADR-0007 round 4 (review findings; awaiting the human grant; no mechanism added):
  - §3: a resume loads into the skeleton re-tied to its shared parameters (`to_empty` unties them, MEASURED); a user
    `build(cfg)` model keeps today's full build, then the load; a node of another sampler fingerprint does not heal a
    CPU-device slot, whose compute class ADR-0007 does not decide (R9).
  - §10: profiling and fit checks hold the slice, not the full model.
- 2026-10-09 — ADR-0007 round 5 (review findings; awaiting the human grant; no mechanism changed):
  - §3: under hf a slot runs no random phase, so those phases' generator states and, without LoRA, the end state are
    taken from the bound record and checked only where the reference builds (ADR-0007 R10, open).
  - §10: each alternative charged its own record (full-walk replay 708 s); the seed checkpoint re-priced as made once
    per recipe, 2222 s per launch at 1 Gbps (was 2884 s with generation on every launch), or 153 s from a per-host
    NVMe cache; the hf trust stated.
- 2026-10-09 — ADR-0007 human decisions 1–6 (awaiting the human grant). §3 row: the init record is bound to the init
  code too; under hf a slot replays its writers' random phases in a scratch layer and checks every state, so the hf
  post-init RNG state is checked (R10 closed by decision 6). §10 row: every ledger transition checked, hf included
  (73 s per 70B hf slot); a record of other init code refused before any draw. Open question 5 decided: the
  init-portability ADR is not commissioned now and returns with the heterogeneous fleet.
- 2026-10-09 — ADR-0008 round 0 (exact MoE; awaiting the human grant):
  - §0: a planned M2c bullet for exact Mixture-of-Experts blocks carried by every existing mechanism.
  - §3: an "MoE block" row (fp32 router scores, stable-sort top-k, one call per (microbatch, expert) on its routed rows
    also when empty, written left folds, byte-move scatters, per-layer aux injected at the layer, no RNG); the compared
    record names each MoE layer's aux hex.
  - §4: INV-1 lists RUNTIME-I18 and MODEL-I8.
  - §8: an M2c row (ADR-0008: MODEL-I8, RUNTIME-I18, SCHEDULER-I21; local E1–E5 first, piece arms with M2b).
  - §9: an MoE paragraph (S1 thresholds ×ρ, device state, pending W, unrouted experts, router precision cost).
  - §10: a decision row; open question 6 (router precision, aux form, aux in the identity record).
- 2026-10-10 — ADR-0008 round 1 (awaiting the human grant):
  - Retargeted from M2c to M4 (the human's "any LLM" scope as the round-1 reviewers report it): §0 bullet, §3 row,
    §8 (the M2c row removed; the M4 row, renamed "hardening and any LLM", carries the MoE work after M2b merges).
  - Stale ids fixed: §4 INV-1 now lists RUNTIME-I18 (round 0's text said I15, M2's resume invariant, although its
    history entry said I18); the §8 row names RUNTIME-I18 and SCHEDULER-I21 instead of I15 and I19.
  - §3 row: the exactness skeleton and the enumerated routing recipe (score, selection key, groups, weights); the
    kernel-vs-written-order reduction rule; aux terms per layer or sequence and z-loss; the selection-bias update and
    its place in the compared state. The compared record names the biases.
  - §9: the bias update's bytes. §10 row: the routing family, the HF boundary rule, v5's hook and fused layout,
    global-batch balancing among the alternatives. Open question 6 widened (aux terms, routing scope, bias update, M4).
- 2026-10-10 — ADR-0008 round 2 (awaiting the human grant):
  - §0: the MoE bullet no longer says "no new mechanism": it names the int64 count lane that carries each
    microbatch's expert counts to the owner of a layer's selection bias, and the HF boundary (HF's objects and names
    kept, the sparse block's forward replaced).
  - §3 row: the router runs in each family's own dtype (compute for Mixtral, Qwen, OLMoE, GPT-OSS, Llama 4; fp32 for
    DeepSeek-V3, Kimi K2), score in fp32; the counts' transport; what the rules buy over HF's eager form, which is
    exact on one CPU map (40/40, MEASURED on a62e4eb): dense gradients for wide stages, CPU/MPS-identical selection
    (`topk` 436/20,000 differ), no accumulate-scatter (refused on MPS under deterministic algorithms).
  - §8 M4 row: the allowlist config map (`head_dim`, `norm_eps`, `rope_style`), the orientation table, checks on real
    config files and against a pinned test-only `transformers`, the HF-object forward replacement with its adapter
    arm blocking closeout until that path exists.
  - §9: the count lane's bytes (`L_m·E·8` per microbatch per peer); the router dtype per family with fp32 as an
    opt-in; the launch term (7.68 vs 4.66 ms per Qwen3 layer, HYPOTHESIS) and S1/S2 conservative for MoE.
  - §10 row: "no new fold, wait or byte" corrected (the count lane); HF's block re-ranked on the measured reasons;
    the HF-modules rule; the grouped GEMM exact under the same call on both paths. Open question 6 adds the loader's
    allowlist, the test-only `transformers` and the HF boundary.
- 2026-10-10 — ADR-0009 round 0 (awaiting the human grant):
  - §0: a planned bullet for any HuggingFace decoder through transformers' own modules (unit windows, eager attention,
    pinned transformers, launch gate, tied tables folded per use on the Head stage).
  - §3: rows "HF modules" and "Tied table across stages"; the ADR-0007 row's hf sentence adds that the HF-modules build
    of a checkpoint draws nothing, so its ledger is the anchor alone.
  - §4: INV-1 lists RUNTIME-I19, MODEL-I9, MODEL-I10 and CONTROL-I24.
  - §5: a "Model code" row (pin, offline, remote code and hub kernels refused, fingerprint; startup only).
  - §8: the M4 row gains the HF-modules work and its ids.
  - §9: an HF-modules paragraph (host time, fp32 cut ×2.43, eager attention memory, tied hop, launch cost).
  - §10: a decision row; open question 7 (ADR-0009's choices and the CLAUDE.md/§3 device-map conflict).
- 2026-10-10 — ADR-0008 round 3 (awaiting the human grant; mechanism frozen, text and numbers corrected against
  ADR-0009, which is now in the tree):
  - §0: the HF boundary is ADR-0009's (Mixtral, Qwen2-MoE and Qwen3-MoE blocks at the pinned transformers); the other
    four families' rows serve from-scratch recipes in M4.
  - §8 M4 row: the HF-object forward replacement at transformers 5.19.0 through ADR-0009's table, hub checkpoints
    once the hub-to-fused byte map has an owner; the allowlist loader only if ADR-0009 is not granted, with
    `transformers` 5.19.0 test-only (the 4.x pin dropped).
  - §9: the HF v5 path's costs (dW in B: t_B ×1.40–1.89, 10.6–59.8% of zero-bubble's reduction kept; fp32 expert
    gradients: S1 thresholds 10,882 / 18,981 / 32,357; routed pending W 0), DERIVED.
  - §10 decision row: the HF sentence names ADR-0009's table and its three classes.
  - §10 question 6: the loader keyed on ADR-0009's grant (the allowlist changes every dense HF RoPE load's bits); the
    routing scope's M4 outcome (the same three checkpoint families either way); the owner of the hub-to-fused map.
- 2026-10-11 — ADR-0008 round 4 (awaiting the human grant; mechanism frozen, text and numbers corrected):
  - §8 M4 row: the experiment list matches the ADR (E1–E4, E7 and E2-HF when ADR-0009 is granted, E5 when it is not;
    E6 on M2b's merged code; MODEL-I4 only if ADR-0009 is not granted); the HF replacement is a class swap to a
    subclass whose own forward is the exact layer, and its adapter arm stays red until ADR-0008 decision 11.
  - §9: unrouted experts `E·(1−k/E)^T ≈ 0` labelled a uniform-routing lower bound; the HF v5 path's +3E launches per
    MoE layer per microbatch over transformers' own block (two f-wide GEMMs for one 2f-wide), 0.12 / 0.90 / 1.92 ms.
  - §10 decision row: HF's block tuned as a rejected alternative (keeps `topk`, banned by the product rule's kernel
    clause; the fp32 combine and fold order are recipe choices traded against HF fidelity); the orientation table
    only if ADR-0009 is not granted; the class swap and "the recipe's bits, not HF's block's".
  - §10 question 6: the HF boundary computes the recipe's bits; how the block passes ADR-0009's trace and classifier.
- 2026-10-11 — ADR-0009 round 1 (awaiting the human grant):
  - §0: tied tables keep the stock fold order (a row-sparse pending fold on the Head stage), rows on the chain
    (replaces "fold per use and live on the Head stage").
  - §3 "HF modules" row: NaN stubs (only the residual crosses a cut, checked), no RNG draw, Linears read outside their
    forward stay unconverted, the gate also on the slot's device. "Tied table across stages" row rewritten: the stock
    order kept by the row-sparse fold (MEASURED equal), stage 0 holds the step's rows, both frames ride the chain.
  - §5 "Model code" row: `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; the fingerprint verifies every transformers and torch file
    against the wheel's RECORD and the loaded functions' code objects (≈ 3.2 s per launch, MEASURED).
  - §8 M4 row: the stock-order tied fold with rows relayed on the chain, the device gate, the RECORD fingerprint, E1–E8.
  - §9: host time 477 vs 480 µs (round 0's 624 vs 858 not reproduced); the tied cost is per-hop bytes plus one
    stage-0↔Head round trip per step, `T_tied` (188 ms at S 4, 348 ms at S 8 for Gemma2-2B, HYPOTHESIS), replacing the
    direct-link 78 ms; the RECORD fingerprint's launch cost.
  - §10 row: 102 of 178 types EXACT, none DIFFER (round 1 census); 8.18e-3 Llama; rejected list adds export-and-split,
    the per-use fold, dense pending tables, a direct link; question 7: `boundary_dtype` refused for HF, the precision
    ADR's two options, tied tables in the stock order with `T_tied`.
- 2026-10-11 — ADR-0008 round 5 (awaiting the human grant; mechanism frozen, text and numbers corrected):
  - §3 MoE row: the stable sort buys a selection defined on ties (`topk`'s tie sets on identical keys), not selection
    identical across backends, whose keys already differ.
  - §8 M4 row: the adapter arm stays red until decision 11 lets ADR-0009's unit trace, module classifier and launch
    gate accept the block (the gate's vanilla reference runs HF's own block).
  - §9: on the HF v5 path each bank's fp32 gradient is held from B to W (5,376 / 1,980 / 2,304 MiB per MoE layer per
    microbatch, ×6.72 / ×7.62 / ×5.76 the native pending W), charged by the planner; "pending W 0" is not a saving.
  - §10 decision row: HF's tuned block restated with the stable sort; its rejection is two MoE functions (fused vs
    separate gate and up GEMMs), traded with HF-block fidelity and 3E–4E launches.
  - §10 question 6: the launch gate joins the trace and classifier, with its options.
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant):
  - §0: tied rows travel in `TIED` frames (replaces "its rows ride the chain").
  - §3 ADR-0007 row: under ADR-0009 buffers come from the run's buffer record (replaces "computed on the CPU by HF's
    own constructors"). "HF modules" row: side inputs on the unit's device; the reference of a device map is the unit
    chain in one process on that map; layer stubs are stride-0 NaN views; buffers are recorded bytes built once on the
    coordinator's CPU (CPU math differs by ISA); the gate checks causality. "Tied table" row: `e_i` sent when B ends,
    the next step's rows after AdamW, both in `TIED` frames forwarded fragment by fragment.
  - §5 "Model code" row: fresh `PYTHONPYCACHEPREFIX`; transformers' RECORD equal on every node, torch's against the
    lock's wheel for the node's platform (mixed-platform fleets kept); code objects equal to the verified source's.
  - §8 M4 row: `TIED` frames, the buffer record, the causality probe, the per-platform fingerprint.
  - §9: the unit window's cost (`c_glue`, stride-0 stubs, 2.7–5.5% in the 80-layer example) replaces "≈ 2%"; eager
    saves 6 B per `b·H·S²` under autocast (3.22 GB at H 32, S 4096; replaces `b·H·S²·2`, 1.07 GB), S cap ≈ 2.4k at 7B
    without recompute; `ΔT_tied` with throughput, warmup, round-trip and scan terms (≈ 269 ms typical, ≈ 495 ms worst
    at Gemma2-2B, S 4, M 8; replaces `T_tied` 188 ms); compile time per launch.
  - §10 row: of the 102 EXACT types, 60 dense in scope, 30 sparse MoE (ADR-0008), 12 encoders refused; rejected list
    adds stage 0 updating the step's rows; V placement named as the planner's after ZB-V. Question 7: `TIED` frames,
    `ΔT_tied`.
- 2026-10-11 — ADR-0009 round 3 (awaiting the human grant; mechanism frozen, text corrected):
  - §2 control plane: the coordinator "never touches numerics, with one exception": for `arch="hf"` it builds the
    buffer record once, at the first launch, under a node's environment guard and an admitted fingerprint (the round-2
    text made it compute those bytes while §2 still said it never touches numerics, and left the builder unguarded).
  - §3 "HF modules" row: a stub value is caught when arithmetic carries it to the logits; a read through a comparison,
    `where`, an index or a cast is not (R13, open; round 2 claimed every other value is NaN). The per-map reference
    exists only for a map whose device classes one host holds (R11, open); vanilla HF is the gate's comparator. The
    buffer record is pinned per tensor in the JobSpec, stored in every wave and never rebuilt.
  - §10: the coordinator row names its one exception; the ADR-0007 row notes that under ADR-0009 an hf checkpoint build
    draws nothing; the ADR-0009 row narrows the NaN claim. Question 7 adds hf LoRA refused, maps no host can reproduce
    (unverifiable or refused), and R13/R14 left open with tied tables at width > 1 refused meanwhile.
- 2026-10-11 — ADR-0009 round 4 (awaiting the human grant; mechanism frozen, text corrected).
  - §2 HF modules row: the reference is vanilla HF in one process on a map of one device class, the unit chain only on
    a mixed map one node holds (R16), any other map refused (R11); rope types that rebuild `inv_freq` in forward are
    refused.
  - §9 HF modules: `ΔT_tied` gains the backward link's burst term; `t_s` includes the tied head GEMM; Gemma2-2B at
    S 4, M 8 is ≈ 197 ms for text and ≈ 414 ms worst against a ≈ 438 ms step (was 269 / 495 / 338).
  - Question 7: maps no node holds refused (recommended), substituted MoE blocks refused until ADR-0008 decision 11
    (R15), host-libm constructor floats (R17) open.
- 2026-10-11 — ADR-0009 round 5 (awaiting the human grant; mechanism frozen, text corrected; `docs/research/adr-0009/round5.md`).
  - §5 "Model code" row: the bytecode prefix is `-X pycache_prefix` on the stage's command line, passed by the worker,
    asserted for every loaded torch and transformers module. Round 4 measured that `PYTHONPYCACHEPREFIX` set in-process
    is ignored but left this row naming it; round 5 measured that a prefix assigned after `import torch` leaves 726 of
    726 torch modules on their in-tree `.pyc` (with `-X pycache_prefix`: 0). The deny-list of fast-path packages becomes
    a refusal of any package transformers probes for (`_is_package_available`) installed outside the `hf` extra's lock
    (mamba's `use_mambapy` switches its scan on an importable `mambapy`); loader variables are refused; torch's locked
    native wheels (`nvidia-*`) join the RECORD check; OS and driver libraries stay outside every RECORD (R18, open).
  - §3 "HF modules" row: the refusal of a map with no oracle holds at every heal (a replacement keeps its slot's
    device class), not only at placement.
  - §10 ADR-0009 row: two tiers are deferred to the precision ADR, not rejected on host time. They win the WAN term
    (94 vs 228 ms per hop per microbatch at 7B on 1 Gbps); a native family computes another function. The 480 vs
    477 µs host times are one contended sample each and decide nothing.
  - Question 7: R18 open; heals keep each slot's device class; the native tier deferred to the precision ADR.
- 2026-10-11 — ADR-0008 round 7 and ADR-0009 round 6 (correction pass under the autonomy switch; every "decision that
  is the human's" in both ADRs took its recommendation, recorded "decided by autonomy, 2026-10-11").
  - §0 ADR-0008 bullet: the block, router and experts classes are swapped and the gate's reference carries the swap
    (the HF-MoE seam rule).
  - §3 HF row: a map no node holds whole is refused at every admission (re-checked), not only at placement and heal.
  - §5 Model code row: `PYTHONPYCACHEPREFIX` removed; the prefix is `-X pycache_prefix`, an interpreter-start fact,
    never assigned in-process; the guard checks rather than sets, so it may follow `hf.py`'s `import torch`, and covers
    `wantrain` modules; importable probed packages refused (unless locked and fingerprinted) and the probe vector
    equal on every node; loaded native objects tied to RECORD files or recorded by sha256 per platform and device
    class (gpu false: macOS and Linux-CPU arms; CUDA libraries at M1).
  - §9 MoE paragraph: the HF v5 bank-gradient hold is multiplied by the microbatches between B and W in the
    zero-bubble order (22.5 / 45.1 GB at 1 / 2 in flight, 4-layer Mixtral stage) and charged in the memory replay.
  - §10 questions 6 and 7: rewritten as the decisions taken by autonomy (ADR-0008 decisions 1–11, ADR-0009 decisions
    1–8) with the risks left open; the CLAUDE.md vs §3 conflict took decision 8's recommendation and stays listed for
    the human.
  - §8 M4 row: the adapter arm is no longer "red until decision 11"; the seam rule closes it, and hub checkpoints
    load through ADR-0009 D7's map.
- 2026-10-11 — the human's decision (~10:35, confirmed ~19:35): the oracle holds per device map (ADR-0005) and, for
  now, only homogeneous fleets (one device class) run; every mixed fleet is refused for every arch at every admission
  until the heterogeneous track returns (CONTROL-I25). Settles ADR-0009 decision 8's recorded conflict.
  - §3 HF modules row, replaced text verbatim: "on a mixed map one node holds, the same unit chain in one process with
    each unit on its map device, so run and reference compute every side input on the same device (a common mode,
    ADR-0009 R16); any other map is refused for `arch="hf"`, re-checked at every admission, whose replacement keeps its
    slot's device class (R11)."
  - §3 Heterogeneous devices row, replaced text verbatim: "Bit-comparable to the oracle *on the same device map*. The
    map is part of the spec, and mixed backends are lossless but not cross-backend identical."
  - §4 INV-1: CONTROL-I25 added, CONTROL-I5 marked suspended.
  - §10 question 7, replaced sentence verbatim: "The conflict (CLAUDE.md: the same bits on every hardware mix; §3: per
    device map) took decision 8's recommendation by autonomy (§3 kept, a map with no oracle refused); it stays listed
    for the human as a recorded conflict."
