# ADR-0007: Slice-only build: ledger replay of the recipe's init program, checked against its init record

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: every slot builds the whole model in fp32 on its CPU and then cuts its slice (`stage.py:130-136`), so a slot
needs `4·P` bytes of host memory at start (ADR-0006 R6): 70B needs 276 GB per slot, 2.2 TB per 8-slot host, and cannot
start. MODEL-I6 asks for a slot that builds only its layers, with every initial bit and the post-init RNG state equal to
the reference's full build. Today nothing checks that a host's initial bytes equal the reference's.

Current bottleneck: init-time host memory `M_init = 4·P` per slot, summed over a host's slots; it exceeds the resume
load (`12·P_s`, `stage.py:747`) whenever `3·P_s,max < P` (S > 3 balanced); `T_step` does not move. Riding on it: every resume pays the full random
init and then overwrites it (`stage.py:132`, then `_resume`), and every slot walks the whole init stream.

Proposed architecture (one decision: how a slot materializes and proves its initial slice; the init bits do not change):
1. *One init program, as data.* `init_phases(cfg)` lists each layer's phases in the recipe's order (construction;
   `gpt2_init` exactly when the build reaches `arch.build_layers`, as today; LoRA), each with its write set.
   `build_layers` runs it phase-major: the reference's bits.
2. *An init record per recipe*, made once in a pinned process: a binding to (config, seed, program id, code id); the RNG state
   before every (phase, layer) and at the end (the *ledger*, 1.12 MB base64 at 70B), anchored at `manual_seed(seed)`; each
   initial tensor's sha256 from the phase-major walk after its last write; the sampler fingerprint.
3. *Ledger replay.* `build_slice` runs only the *writers* of the slot's home tensors (its layers, and the other end of a
   tied table), each phase from its ledger state, checking the state after it (hf: into scratch, R10). Each home tensor is finalized once,
   after its last writer: hashed against the record, trimmed to ADR-0006 pieces, stored. Then the generator is set to
   the ledger's end state.
4. *A resume never draws:* skeleton slice `to_empty` on the device, re-tied from the skeleton's sharing map (`to_empty`
   unties, MEASURED), every tensor and the RNG state from the wave; a wave without the RNG state is refused. PROFILE and
   FIT take the same slice, privately filled. `arch="user"` keeps today's full build, then the load.
5. *Fail closed, one exclusion rule:* a node whose fingerprint (from a pinned child process at HELLO) or whose slot's
   transition or digest check differs from the record is excluded from init for that record; selection reruns. A wrong
   binding or anchor refuses the job once. A CPU-device slot's heal keeps the fingerprint as a compute-class proxy (R9).

First-principles reasoning: the mt19937 state at a call depends only on the draws before it, and a draw only on
(state, op, shape, dtype, strides): the state before each (phase, layer) is a recipe constant, and a layer rebuilt from
it has the full build's bytes. Only the reference's bytes tell a foreign sampler apart, so the recipe carries their
digests from another traversal. MEASURED: 84 replay and 168 ledger slice builds equal the full build, RNG included.

Mathematical justification: init host peak per slot `4·P → 4·(P_layer,max + P_held)` (+ `4·P_s` when the home is
the host); the resume load (`12·P_s`, unchanged) then binds the host; `T_init` per slot `P → P_writers` draws plus
hashing and copying `4·P_s`; `T_R` loses `T_walk`; PROFILE/FIT `4·P → 4·P_s`; network 1.2–3.1 MB; `T_step` unchanged.

Expected improvement (DERIVED, `cost_r3.py`; 70B/405B times HYPOTHESIS, from one-run sampler rates, until E4): 70B S 32,
8 slots on 80 GB devices: 2207 → 207 GB per host with the resume load (27.4 GB for the build; device split pending), so
it starts and heals on 256 GB hosts; host home (gpu false), S 8: build 2207 → 303 GB, `16·P_s` binds. Init per launch
70B S 8 588 → 94 s after an 811 s record once per recipe; each resume saves `T_walk`: 9 / 588 / 3463 s (7B/70B/405B).

Alternatives (§Alternatives): do nothing; full-walk replay; recipe v2 (torch, portable); seed checkpoint; interception.
Risks: R1 samplers may differ by CPU class (checked); R2 a layer above free DRAM; R3 user models keep the full build; R4
the record grows with P; R5 a first launch is 302 s slower at 70B; decided: R7 the resume load binds host memory, R8 rollback
by code revert, R10 hf replays its random phases into scratch and the binding holds the code id; open: R9 a CPU-device slot's compute class.

Falsification criteria: one differing sha256 of an owned tensor or the RNG state against the full build or the record,
or of the identity manifest against the oracle; a build, resume or PROFILE host peak over its bound; a slice build or
record > 10% over its model; a resume that draws; a foreign fingerprint, a wrong byte or a bad or missing record not refused.
Local experiment: E1–E5, CPU only (gpu false), localhost clusters; E5(g2), E-a only with a second ATen capability.
Estimated Vast.ai cost: none: local only (the device-home split on a real GPU stays pending).
Recommendation: PROCEED (local experiment first). No change of the init bits; recipe v2 is not built.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0007-slice-only-build.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

MODEL-I6 (DEFERRED-TO: M4): a stage builds only its slice `[a, b)`, identical to slicing the full build; the problem
statement doubts this is possible without a per-parameter generator and asks that a reference change be priced. Hard
constraints: the product rule; fp32 masters; ADR-0006 (pieces, RUNTIME-I12 logical checkpoints, resume at another
width) and heal; one decision; ≤ 400 lines; gpu false. Derivations: `docs/research/adr-0007/derivation.md`.

## Current Architecture

FACTS (tree 318422c, read):
- `StageRuntime.__init__`: `setup_determinism(seed)`, `full = build_layers(spec.model, seed)`, `ModuleList(full[a:b])
  .to(device)` (`stage.py:130-136`). The same full build: `resources/fit.py:55`, `resources/profile.py:55`,
  `validation/state.py:105` (names only), `validation/reference.py:36` (the oracle).
- Init draws from the one global CPU generator: `torch.manual_seed` at `gpt.py:372` (builtin), `arch.py:227` (general:
  construction, then `gpt2_init` redraws every Linear weight and Embedding in traversal order, `arch.py:207-220`),
  `gpt.py:426` (user `build(cfg)`, one call for the whole list). `arch.build_layers`, the only caller of `gpt2_init`, is
  reached only under `arch="hf"` or when `_uses_general_arch(cfg)` (`gpt.py:368-376`), which ignores `init_scheme`
  (`gpt.py:433-442`): a builtin-shaped config that declares `gpt2` draws torch defaults today. `hf.py:347` runs
  `arch.build_layers` (random), then `load_into(strict=True)` overwrites it; LoRA is applied last (`gpt.py:345-348`).
- A tied Head's `out.weight` is the Embed's `tok.weight` (`arch.py:191`); under `gpt2` the last write to that table is
  the Head's `normal_` (`arch.py:211-220`). Tying without `gpt2` is refused (`gpt.py:124-130`).
- LoRA (`lora.py:108-110`) seeds each adapter from `(seed, global layer index, name)`, but `LoRALinear.__init__` builds
  `nn.Linear`s (`lora.py:41-42`) that draw from the global generator first. `wgrad.convert` (`wgrad.py:127-142`) leaves
  a shared-weight Linear (a tied Head) as `nn.Linear`. No builtin, general or hf module holds a buffer (`grep`).
- The RNG state is compared state: checkpointed (`stage.py:721`), restored only when present (`stage.py:755-756`),
  hashed by the manifest (`state.py:91`). Nothing in training draws from it. `_resume` loads the slot's whole payload
  (fp32 weights, AdamW `m` and `v`: `12·P_s`) to the CPU (`stage.py:715-718, 747`).
- `setup_determinism` runs single-threaded and pins x86 to AVX2 "before the first kernel dispatch"; ARM is not pinned
  (`determinism.py:17-24`): this arm64 host has one capability. `probe.measure` runs matmuls unpinned (`probe.py:79-133`).

## Current Bottleneck

`M_init(host) = Σ_{slots} (4·P + B_0)`: 7B (h 4096, 32 blocks, V 32k) 26.9 GB per slot; 70B (h 8192, 80 blocks, GQA 8,
d_ff 28672) 275.9 GB; 405B (h 16384, 126 blocks, V 128k) 1623 GB (`cost_r0.py`). Riding: `T_R` and every launch pay `T_walk`.

## Evidence

Files, sha256, commands and verbatim output: `derivation.md` §1 table (macOS arm64, torch 2.14.1, ATen DEFAULT, 1 thread).
- MEASURED (`probe_r1.py`): replay equals the full build for all 84 contiguous slices of builtin, general default, `gpt2`,
  `gpt2` tied with GQA and RoPE (sha256, RNG end state); C1–C4 caught; 168 ledger builds equal; a state 5056 bytes;
  fingerprint 1.6–2.0 ms; sha256 2.3–2.5 GB/s; no record built (E1). Round 0, one run: meta construction draws nothing;
  `normal_` 7.32 ns, `uniform_` 1.18 ns per element. `probe_r4.py`: `to_empty` unties a tied pair, the re-tie restores
  it; builtin and general layers hold 0 buffers. DERIVED: `cost_r1`–`cost_r3.py`, `cost_r5.py` (round 5, §10).
- Source read (HYPOTHESIS for our hosts): ATen's CPU `normal_` is AVX2 polynomials under `CPU_CAPABILITY_AVX2`, libm
  elsewhere, a serial double path below 16 elements. Cited: DeepSpeed `zero.Init`; Megatron; torchtitan seed
  checkpoint; FSDP; JAX threefry; Random123; Haramoto 2008; L'Ecuyer 2017; in-toto; BLAKE3.

## Hypothesis

- **H1, exactness.** `build_slice` gives every owned tensor the full build's bytes and leaves `get_rng_state()` equal to
  the full build's, for builtin (also declaring `gpt2`), general (default and gpt2), tied, GQA, LoRA and hf recipes,
  every `[a, b)`, after the trim to ADR-0006 pieces at any w; `init_record`'s digests and ledger equal those of
  `build_layers`; the new `build_layers` equals the legacy fixture for every preset and for a builtin-shaped `gpt2` config.
- **H2, memory.** Host peak per slot during the build ≤ `4·(P_layer,max + P_held) + B_0`, and during a resume ≤
  `12·P_s + B_0`; each plus `4·P_s` when the home is the host. A PROFILE or FIT subprocess holds its slice, not `4·P`.
- **H3, time.** `T_build(slice)`, `T_rec` ≤ 1.10 × their models with every rate measured on the host; a resume draws nothing.
- **H4, end-to-end.** With `build: "slice"` the identity manifest equals the oracle's on every arm, healed before and after the first wave.

## First-Principles Derivation

**What a slice must reproduce.** (i) The bytes of every tensor its layers reference. (ii) The generator's state after
init (compared state, `state.py:91`; nothing later moves it). (iii) The full model's structure where it changes
computation (`convert` classes, MODEL-I3; trainable flags; names; sharing): from `skeleton(cfg)`, the full model on the
meta device (no storage, no draw, MEASURED); a slice computing it from its own modules gets a tied Head wrong.

**The premise is false.** mt19937 is a sequential integer generator; a call's output depends on the state at the call
and on (op, numel, dtype, strides). By induction, if every call before position k runs with the same arguments, the
state at k is the same. So the state `ledger[p][i]` before phase p of layer i is a constant of the recipe.

**Home tensors, writers, finalize.** `H` = every tensor an owned layer references (by identity in the skeleton: a tied
table is one tensor). A *writer* is any layer whose write set names a tensor of `H`, owned or not: the Embed and the
Head both write the tied table, and under `gpt2` the Head's write is last. Each writer's phase runs in its CPU working
layer, so each tensor of `H` receives the full build's writes in the full build's order, and its final bytes are equal.
A tensor is *final* only after its last writer: it is hashed, trimmed and stored then, and not at its owner's last
phase. A stage holding one end of a tied table keeps it on the CPU until the Head's last phase (`P_held = V·h`, 1.05 GB
at 70B; derivation §7.3).

**Ledger replay.** A slot runs only its writers: per writer, per phase, `set_rng_state(ledger[p][i])`, the phase, then
a check that the state equals the next ledger entry. Phases run layer-major on the module the constructor built, as
`build_layers` does. No init call runs on a device (CUDA's output depends on the SM count). Only the public
`set_rng_state`/`get_rng_state` the checkpoint uses; no draw-count rule, no private state layout.

**Binding and anchor (derivation §7.1).** The record's binding is sha256(cfg with `hf_model_sha256` and LoRA fields,
seed, program id = sha256 of `init_phases(cfg)`, code id = sha256 of the init-path sources `src/wantrain/model/{gpt,arch,lora,hf,wgrad}.py`,
`torch.__version__` and `torch.version.git_version`); a slot refuses another (`INIT-RECORD`) before any draw, so a record of another seed,
config, init program or init code is refused in O(1), µs. Every slot also checks `ledger[first]` against `manual_seed(seed)`; the slots
partition the layers, so a started run, hf included (its random phases replay into scratch, below), has checked every transition to `end`.

**Digests from another traversal (derivation §7.2).** `init_record` is one phase-major walk holding one working layer:
construction on the real constructor, every later phase on the layer's skeleton `to_empty` and NaN-filled. After each
phase its write set must hold no NaN and every other tensor must still be all NaN (`INIT-PROGRAM`): the write sets are
proved and no phase reads a tensor (R6). Each tensor is hashed right after its last writer. The slots replay
layer-major on constructed modules: a drift of either traversal fails closed (`INIT-DIGEST`); one common to both is
caught where `build_layers` runs (E1; `reference.py`, which also checks every ledger state).

**Paths.** builtin and general: as above; the `gpt2_init` phase is in the program exactly when the build dispatches
to `arch.build_layers` and `init_scheme == "gpt2"`, so a builtin-shaped config that declares `gpt2` has one phase, as
today (derivation §8.1). LoRA: a third phase, `lora.apply`'s per-layer wrapping primitive
(`first_index=i`, no empty-match error; the "matched no Linear" refusal and `WT_LORA` stay whole-program, from the
skeleton, §10.3); its `nn.Linear` constructor draws are in the ledger. hf: the checkpoint overwrites every tensor
(MODEL-I4), so a slot runs its writers' random phases in one reused scratch working layer (≤ `4·P_layer,max`, already
budgeted), checks every transition and discards the values; then skeleton slice `to_empty`, strict span load, LoRA from
its ledger entry, the end state (+73 s per 70B slot, decision 6; §7.6, §10.1). A tied table's writer is the load
of `embed_tokens`, read alone by a Head-only stage; every hf home tensor meets its digest (§9.4). user: keeps the full build (R3).

**Resume and heal.** A resume loads every tensor and the RNG state from the wave, so initial values are irrelevant: the
skeleton slice `to_empty` on the device, re-tied from the skeleton's sharing map (`to_empty` gives each referrer its own
Parameter, MEASURED, derivation §9.1), then the load (RUNTIME-I12 at any width). Exact only if every tensor is in the
wave: true for builtin, general and hf (no buffers, MEASURED); `arch="user"` may hold non-persistent state, so it keeps
today's full build, then the load (§9.2). A wave without the RNG state or a tensor is refused naming the file (today
`_resume` keeps the state when `rng` is absent, `stage.py:755`). PROFILE and FIT take the same slice, privately filled
(seed 0; CPU time measured in E4, §9.5, §10.6). A relaunch before the first wave reruns step 0: recomputing lost work.

**The init class (constraint, proved; derivation §6).** Scheduling, routing and retries do not change the function
`f_c` a CPU class c computes for a sampling call, and without information about the reference's bytes a fleet of one
foreign class is indistinguishable from the reference's. Hence the digests: the sufficient check. The fingerprint (a
fixed probe over every sampler path init calls, plus `get_cpu_capability()`) is a necessary condition checked in 2 ms,
valid only from a process pinned by `setup_determinism` before its first dispatch, so HELLO takes it from a pinned child
(§7.4); a declared tuple is a diagnostic. Admitting foreign classes at step 0 needs shipped bytes or a portable sampler
(R1). After the first wave a CPU-device slot still computes with its CPU class, so its heal keeps the fingerprint as a
necessary proxy (R9, §9.3). **One exclusion rule,
not an admission policy:** a node whose fingerprint differs, or whose slot raises `INIT-CLASS`, a transition's
`INIT-LEDGER` or `INIT-DIGEST`, is excluded from init for that record and selection reruns; selection and
`find_replacement` apply it from HELLO. `INIT-RECORD` and an anchor `INIT-LEDGER` are the record's fault, equal on every
node: the job is refused once, naming the record (§10.2).

## Mathematical Model

Only these terms move: `T_total = T_init + Σ_steps T_step + N_R·T_R`, with host memory `M(host)` as the admission
constraint. Variables: P, `P_s`, `P_writers` (`P_s` plus a tied table's other end), `P_layer,max`, `P_held` (`V·h` on a
tied end stage), k slots per host, `t_u` 1.18 ns, `t_n` 7.32 ns (MEASURED, one run; E4), `r_sha` 2.3 GB/s (MEASURED); `r_mem` 5–10 GB/s per pass,
`c_layer` 1 ms, `T_skel` 1 s (HYPOTHESIS, E4); `h_s = 4·P_s` when the slot's home is the host (gpu false), else 0.

```
today:     M_slot = max(4·P, 12·P_s + h_s) + B_0                     T_init,slot = T_walk
proposed:  M_slot = max(4·(P_layer,max + P_held) + h_s, 12·P_s + h_s) + B_0     (resume: 12·P_s, unchanged)
           T_walk  = P_lin·(t_u + t_n) + P_emb·2·t_n  (gpt2 runs: general path)   P_lin·t_u + P_emb·t_n  (otherwise)
           T_slice = T_walk·P_writers/P + 4·P_s/r_sha + 4·P_s/r_mem + T_skel        per launch before the first wave
           T_rec   = T_walk + 4·P/r_sha + (n_fill + n_scan)·4·P/r_mem + c_layer·n_phase·(L+2)    once per recipe
           T_R     = T_walk + T_load  →  T_load                    network: record 1.2–3.1 MB per slot, once
           M_prof  = 4·P → 4·P_s  (+ its gradients and activations) per PROFILE or FIT subprocess (user: 4·P)
```

`n_fill = n_phase − 1`, `n_scan = n_phase` (MODEL-I6's NaN fill and scans): 15% of round 2's `T_rec` at 8 GB/s
(derivation §8.2). Per 8-slot host (`B_0` excluded; 70B and 405B times HYPOTHESIS until E4, §10.6):

| Model, plan (scheme run) | today | device home: build / with resume | host home: build / with resume | T_walk | T_slice | T_rec |
|---|---|---|---|---|---|---|
| 7B, S 8 (builtin: torch defaults) | 215 GB | 6.4 / 80.7 GB | 33.3 / 108 GB | 9 s | 4 s | 24 s |
| 70B, S 8 (gpt2), init only | 2207 GB | 27.4 / 828 GB | 303 / 1104 GB | 588 s | 94 s | 811 s |
| 70B, S 32 (gpt2) | 2207 GB | 27.4 / 207 GB | 96 / 276 GB | 588 s | 24 s | 811 s |
| 405B, S 16 (gpt2) | 12987 GB | 102 / 2435 GB | 914 / 3247 GB | 3463 s | 274 s | 4778 s |

(Untied; a tied end stage adds `4·V·h` = 0.52 / 1.05 / 8.41 GB. 70B S 8 trains nowhere: `16·P_s` = 138 GB per slot.)
`T_walk` = 17 / 69 / 553 steps at N 32 / 128 / 1024 (`(T_walk/P)·N·F_eff/(6·T_tok)`, `F_eff` 4e14, `T_tok` 2^20, HYPOTHESIS).

**What moves, and when the gain disappears.** The init term of host memory falls by `k·4·(P − P_layer,max − P_held)`;
the host bound becomes `k·max(build, 12·P_s + h_s)`: a plan gains admission only where `k·(12·P_s + h_s)` fits and
`k·4·P` does not (70B, 256 GB hosts, device homes: S ≥ 26 at k 8; host home: `16·P_s` binds, S 32 k ≤ 7). The build
stops fitting when `4·P_layer,max` exceeds free DRAM (1T, R2). Init per launch falls from `T_walk` to `T_slice`; the
gain disappears for a recipe launched once (905 s against a 603 s full walk at 70B S 8, R5) and in runs over 100 × `T_walk`.

## Expected Improvement

As stated at the top (DERIVED, `cost_r3.py`; device split and 70B/405B times HYPOTHESIS): host 2207 → 207 GB at 70B S 32
k 8 (device homes) or → 303 GB at S 8 (host home); PROFILE/FIT `4·P → 4·P_s`; 588 → 94 s per launch; `T_step` unchanged.

## Alternatives

One init-only point (`cost_r5.py` §A): 70B, S 8, w 1, k 8, one 256 GB host, 1 Gbps, NVMe 2 GB/s (HYPOTHESIS). Every row
carries its own digest record (per launch −15 s without digests; then only the ledger still pays once, its 588 s walk);
host DRAM is the build peak with device homes; all share the resume load (828 GB).

| Option | Host DRAM | Per launch (with digests) | Once per recipe | Recipe change | Mixed CPU classes | Why not chosen |
|---|---|---|---|---|---|---|
| Today | 2207 GB | cannot start | 0 | no | silent | MODEL-I6 unmet |
| Do nothing, strongest: one file-backed full build per host, host lock | page cache, 276 GB disk | 879 s | 708 s | no | as chosen | 276 GB disk per host (1623 GB at 405B); 9× per launch |
| Seed checkpoint, made once, shipped per launch (NVMe-cached: 153 s, 276 GB disk per host) | 27.4 GB | 2222 s, 276 GB over the link | 708 s | no | admitted | 24× per launch on WAN (cached: 1.6×); its value is R1 |
| Per-tensor interception, full walk | 8.4 GB | 608 s | 708 s | no | as chosen | a missed op is a silent bit; gains only past R2 |
| Full-walk replay (round 0) | 27.4 GB | 608 s | 708 s | no | as chosen | 514 s slower per launch than the ledger |
| **Ledger replay (chosen)** | 27.4 GB | 94 s | 811 s (ledger and digests) | no | excluded; heal device slots after the first wave | — |
| Jump-ahead or draw-count advance | 27.4 GB | ≈ 94 s | 811 s | no | as chosen | same time with GF(2) code or ATen's private rule |
| Recipe v2, torch per-parameter generator | 27.4 GB | 94 s | 101 s (8 cores) | yes | as chosen | 710 s once per recipe for a new recipe identity; not portable |
| Recipe v2, portable counter-based sampler (3 / 20 / 50 / 200 ns per sample, HYPOTHESIS) | 27.4 GB | 46 / 193 / 451 / 1745 s | 54–1752 s (8 cores) | yes, plus a native sampler | admitted if portable | its value is R1, unmeasured until E-a; init-portability ADR |

## Trade-offs

A record per recipe and init code (811 s at 70B; first launch +302 s; a change of an init-path source or of torch needs a new record);
an hf slot pays its 73 s random walk (70B); a tied end stage holds the table on the CPU; user models keep the full build; only in-fingerprint nodes init or heal a CPU-device slot; a pinned child per agent start (≈ 1–2 s).

## Risks

- **R1 (OPEN, pre-existing, now checked):** CPU samplers may differ by class (source read); today a mixed fleet diverges
  silently, now a foreign node is excluded and wrong bytes refused, until the init-portability ADR (returns with the heterogeneous fleet, decision 3; E-a; §6).
- **R2:** `4·P_layer,max` binds at about 1T (a 31 GB block); per-tensor interception is the follow-up, on `MEM-INIT`.
- **R3:** user models keep `4·P` and today's resume. A per-layer `build_layer(cfg, i)` contract is a follow-up.
- **R4, R5:** `T_rec ∝ P` once per recipe (4778 s at 405B, one core, sequential); a recipe launched once pays 905 s
  against a 603 s full walk (70B S 8). **R6 (refused):** a phase that reads a tensor fails `INIT-PROGRAM`; none today.
- **R7 (DECIDED, 4):** the resume load, `12·P_s` per slot (`stage.py:747`), is the host bound (828 GB at 70B S 8,
  207 GB at S 32, k 8); MEM-INIT charges `max(build, resume)` (SCHEDULER-I18); a per-tensor resume load: follow-up.
- **R8 (DECIDED, 5):** `build: "full"` rolls back only the step-0 build; the draw-free resume, the RNG-less refusal
  and the phase-major `build_layers` roll back by code revert (E1 fixture, E2 heals under both values); no `resume` field.
- **R9 (OPEN, pre-existing; round 4):** with gpu false every slot computes on its CPU, and a foreign CPU class runs other
  GEMM and libm kernels; `find_replacement` never matches the CPU class. Here a CPU-device slot's heal in a run with a
  record requires the record's fingerprint (necessary, not sufficient); a compute-class check is a new mechanism.
- **R10 (CLOSED by decision 6; round 5):** hf post-anchor states were trusted from the record. Now each hf slot replays its writers' random
  phases into scratch, checking every transition (+73 s per 70B slot), and the binding holds the code id (µs): other init code is `INIT-RECORD`.
- **Refuted:** a tied default-init Head-only stage (1); "two decisions in one grant", in part (2); none in rounds 3–5.
- Security: the record (JSON, no pickle, base64 5056-byte states, skeleton names, ≤ `cap(cfg)`, §10.4) is validated and bound before
  any draw; on every arch, hf included, a tampered record fails closed (the slots partition the layers and re-check every transition,
  `end` and every digest; the anchor is the first state). Cost: a pinned child per agent start, 15 s of hashing per 70B S 8 first
  launch, a µs re-tie and code id; 0% of `T_step`. `reference.py` refuses another fingerprint.

## Falsification Criteria

- F1 (H1): any owned tensor's sha256 or the RNG state differs from the legacy fixture's `[a:b]` or from the record, in
  any arm, traversal or process (`PYTHONHASHSEED` 0, 1, random); a record's digest or ledger entry differs from
  `build_layers`'; the new `build_layers` differs from the fixture for any preset or the builtin-shaped `gpt2` config; a
  materialized skeleton slice with more than one Parameter per shared group. One differing bit refutes it.
- F2 (H4): any sha256 difference of the identity manifest (weights, m, v, step, RNG, every loss hex) against
  `FACTORY_REFERENCE=1` with `build: "slice"` in E2, including heal before and after the first wave.
- F3 (H2): host peak RSS above `4·(P_s + P_layer,max + P_held) + B_0` during the build, or above `16·P_s + B_0` during
  a resume (gpu false), by more than 10% in E3; a PROFILE or forced FIT subprocess that saves less than 90% of
  `4·(P − P_s)` against today's of the same stage; a `build_layers` call in `profile.py` or `fit.py` outside `build:
  "full"` and `arch="user"`.
- F4 (H3): median `T_build(slice)` or `T_rec` above 1.10 × its completed model over 5 runs in E4, with every rate
  measured in E4; or a resume whose generator state before its load differs from the state `manual_seed(seed)` leaves.
- F5: a node of another fingerprint selected for init, or healing a CPU-device slot of a run with a record; a HELLO
  fingerprint not from a pinned child (E5(g1)), or, on a host with two ATen capabilities, unlike the slot's (E5(g2)); a
  slice launch without a record, or with one of another seed, config, program or code id, that starts; a matching fingerprint
  with a wrong byte, a perturbed ledger entry or `end` of any arm (hf included) or a wave without RNG that is not refused; a binding or
  anchor failure that excludes a node instead of refusing the job.

## Local Experiment

CPU only (gpu false); every timing SIMULATED on this host.
- **E1, exactness (unit, separate processes).** n_layer 4 and 6; builtin, and builtin-shaped declaring `gpt2` (one
  phase, equal to the fixture); general default and gpt2; tied gpt2 ([0,1), [L−1,L)); GQA; SwiGLU, RMSNorm, RoPE; LoRA
  (`attn`, `all`; Embed-only and Head-only stages); hf (tiny local checkpoint; tied `[L+1, L+2)`, `[k, L+2)`); every
  `[a, b)`, pieces at w 2, 3, 4. Per-tensor sha256, `convert` classes, trainable flags, RNG state against fixture and
  record; `init_record` against `build_layers`; new `build_layers` vs fixture per preset; no buffers; an S 1 tied slice
  has one Parameter per shared group. Controls that must FAIL: C1–C4; the tied table finalized early; names picking the
  unscaled std; a real tie target; `convert` on the slice; LoRA by local index; a non-contiguous working tensor; a
  perturbed ledger entry (every arm, hf random phases and hf `end` without LoRA included); a record of another code id;
  a reading phase; `gpt2` from the declared field; a Head-only hf stage without the `embed_tokens` read; `to_empty`
  without the re-tie.
- **E2, identity and chaos.** Localhost clusters S 1–4 (S 1 tied), a `zero1` wide stage at w 2, bf16 and fp32, tied
  and untied, `build: "slice"` against the oracle by sha256; heal and restart before and after the first wave, under
  both `build` values; a wave stripped of `rng` refused; a user model with a non-persistent buffer healed after the
  first wave (full-build resume); resume at another width once RUNTIME-I12 exists (pending until M2b).
- **E3, memory.** A ~250M-parameter general config (h 1024, 16 blocks, V 32k), S 4, tied and untied, four slots on
  localhost: `ru_maxrss` per slot during the build and during a resume against F3; `B_0` measured as a 0-layer build;
  PROFILE and FIT (`force=True`) per subprocess against today's; the `build_layers` structural check.
- **E4, time.** `T_build(slice)`, `T_rec` and the full walk, 5 runs, single thread; `t_u`, `t_n` (5 runs at 2^28
  elements, ≥ the largest 70B tensor), `r_sha`, and `r_mem` for the NaN fill, the scan and the trim copy, `c_layer`,
  `T_skel`; draws during a resume (F4); PROFILE of one stage, private fill against built values (a gap > 10% reopens the fill).
- **E5, init class.** (a) in-class nodes too few: refused with per-fingerprint counts; enough: launched on them only;
  (b) a uniform foreign fleet refused; (c) one flipped bit of one draw under a matching fingerprint: `INIT-DIGEST`;
  (d) no record: `INIT-RECORD`; (e) a foreign spare excluded from a heal before the first wave, and after it from a
  CPU-device slot's heal (R9; a device slot taking one: N/A while gpu is false); (f) a foreign reference refused;
  (g1, here) HELLO's fingerprint comes from a child that calls `setup_determinism` before any dispatching import
  (WOULD-FAIL-IF: computed in the agent process); (g2) an unpinned agent with ATen forced to another capability gives
  the pinned slot's fingerprint: N/A here, pending where `avx2`/`avx512` exist (§8.4); (h) a seed-43 or
  other-`init_std` record: `INIT-RECORD` before any draw; a first ledger entry not `manual_seed(seed)`'s: `INIT-LEDGER`;
  either refuses the job once and excludes no node.
- **E-a, ISA probe (informative for R1).** Fingerprint and a small record on macOS arm64, linux/arm64 and linux/amd64
  (`default`, `avx2`) containers if Docker exists, emulation labelled; else pending.

## Vast.ai Experiment

None while gpu is false; with GPUs on, a PLAN.json measures the device-home split (no bits depend on it).

## Migration Impact

No change of the init bits or the reference's values; existing checkpoints keep their bytes and format. Behaviour
changes: a resume refuses a wave without the RNG state (`stage.py:721`, the only writer, always writes it); PROFILE and
FIT measure a privately filled slice; a CPU-device slot of a run with a record heals only on its fingerprint;
`arch="user"` keeps today's build, resume, PROFILE, FIT and manifest. New `JobSpec.build: "slice" | "full"` (`"full"`
until E1–E5 pass) and `JobSpec.init_record`. Rollback: `build: "full"` for the step-0 build, the rest by code revert (R8). Workstreams: model,
runtime, scheduler, control.

## Validation Plan

MODEL-I6, RUNTIME-I14, SCHEDULER-I18, CONTROL-I19 on the unit (E1, E5), integration, identity and chaos (E2) rungs; E3, E4 pinned.

## Ideas considered

Rounds 0–4, each idea with verdict, reason and number: derivation §11 (moved verbatim; adopted: ledger replay, writers,
digests, bound record, pinned fingerprint, draw-free resume, re-tie). Round 5, none adopted as a mechanism (round-3 freeze):

| Idea (lens) | Verdict | Reason and number |
|---|---|---|
| Code identity (init-path sources, torch version and build hash) in the binding (frontier) | ADOPTED, decision 6 | µs per launch, 0% of `T_step`; closes code drift on every arch; a non-hf drift fails in O(1), not after `T_slice` (94 s) |
| hf slots replay their writers' random phases into one reused scratch buffer (first-principles) | ADOPTED, decision 6 | 73 s per 70B slot (the §7.6 saving given back), ÷w if one slot per stage; makes the hf chain checked |
| Per skipped hf phase a draw-count digest; hf's RNG state derived from anchor and LoRA only (consistency) | FOLLOW-UP / init-portability ADR | the first needs ATen's draw-count rule (rejected, round 1); the second is a recipe change |
| `INIT-RECORD` and the anchor checked at admission in the control plane (first-principles) | FOLLOW-UP | this round refuses the job on the first such report (§10.2); admission saves one launch, a new placement of the check |
| Binary ledger sidecar bound by sha256 (alternatives-and-risk) | FOLLOW-UP | base64 with `cap(cfg)` (§10.4) removes the edge: 3.05 MB under 3.11 MB at 405B with LoRA; a sidecar saves 25% of ≤ 3.1 MB once |
| NVMe-cached seed checkpoint as a baseline row (alternatives-and-risk) | Priced (row), no mechanism | 153 s per launch against 94 s, 276 GB of disk per host, one 2208 s transfer per host (`cost_r5.py` §A) |
| PROFILE/FIT fill: norms ones, linears within ±init_std (alternatives-and-risk) | FOLLOW-UP, on E4 | timing only, no bit; reopened if the private fill moves PROFILE time > 10% |
| No fingerprint exclusion for an hf recipe without LoRA (consistency) | REJECTED by decision 6 | an hf slot now replays its random phases into scratch, so its class matters as on every arch; CPU-device slots stay under R9 |
| Record metadata in a sidecar beside the identity manifest (consistency) | FOLLOW-UP | HYPOTHESIS: the run and its reference carry the same `JobSpec.init_record`, so the fields agree; a metadata-only gap fails E2 visibly, it cannot hide a bit |
| SVE256 is a second aarch64 ATen capability; hash the probe, not the string, or pin ARM (frontier) | FOLLOW-UP (R1, R9) | moves who is excluded, no bit; E5(g2) becomes runnable on an arm64 SVE host |
| Below 16 elements and non-contiguous, `normal_` uses double libm on every host; glibc 2.43 CORE-MATH omits `logf`/`sinf`/`cosf` (frontier) | DEFERRED to the init-portability ADR | narrows R1: the 15-element probe is load-bearing on pinned x86 |
| `rng_philox.py`, a CPU reimplementation of CUDA Philox (frontier) | DEFERRED to the init-portability ADR | evidence for the portable-sampler row (3–200 ns per sample, HYPOTHESIS) |

## Changelog

- Rounds 0–4: derivation §11.
- Round 5 (2026-10-09): 2 majors (one defect: the hf chain was claimed checked) and 8 minors, all valid, no mechanism
  changed: R10 and question 6; security, F5, E1 scoped to non-hf with an hf expected-failure pin; binding or anchor
  failures refuse the job; per-layer LoRA primitive; base64 record and `cap(cfg)`; alternatives each charged their own
  record, seed checkpoint re-priced (2884 → 2222 s); `S > 3`; one-run rates labelled; PROFILE fill measured; user manifest.
  Human decisions 1–6 (2026-10-09, §Recommendation): decision 6 adds hf scratch replay and the code id, closing R10's 2 majors.

## Amendments (texts in DESIGN.md and the specs, each in its -history.md and `freeze.mjs amend --by ADR-0007`)

DESIGN.md §3, §8 (M4), §10 (row, question 5); SPEC-model §2, §3, MODEL-I6, task 1, §6; SPEC-runtime RUNTIME-I14, task
4; SPEC-scheduler SCHEDULER-I18; SPEC-control CONTROL-I19. Round 5: DESIGN §3 row and §10 row; MODEL-I6 (LoRA, hf,
record encoding); RUNTIME-I14 (cap, report routing), task 4; CONTROL-I19 (a binding or anchor failure refuses the job).

## Recommendation

PROCEED with the slice-only build by ledger replay of the unchanged init program, checked against the recipe's bound
init record; draw-free resume; one exclusion rule for init. E1–E5 first (E-a informative). Do not build recipe v2 now.

Human decisions (2026-10-09): (1) yes, a `build: "slice"` launch needs the recipe's init record, made once on a host of the class
the recipe is to use; (2) yes, until the init-portability ADR a node of another sampler fingerprint does not generate init at a first
launch and heals no CPU-device slot (R9); (3) not now: that ADR returns with the heterogeneous fleet (DESIGN q. 5); (4) yes, the resume
load is the host bound, a per-tensor resume load is a follow-up (R7); (5) rollback by code revert, no `resume` field (R8); (6) hf replays
its random phases into scratch (73 s per 70B slot) and the binding holds the code id (µs): R10 closed. Implementation after M1.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
