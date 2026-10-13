# DESIGN: wantrain, synchronous lossless pipeline training over WAN on heterogeneous hardware

**Status: PROPOSED — awaits a human grant** (`node agents-architetura/tools/factory/approve.mjs grant docs/design/DESIGN.md`).
**Records the design of the engine merged in `9f7d81c` (milestone M0, done) and the milestones after it.**

This document is the design of record. Workstream specs: [SPEC-runtime](SPEC-runtime.md), [SPEC-model](SPEC-model.md),
[SPEC-scheduler](SPEC-scheduler.md), [SPEC-transport](SPEC-transport.md), [SPEC-control](SPEC-control.md). What exists:
[CURRENT_ARCHITECTURE](CURRENT_ARCHITECTURE.md), [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md). Threats:
[THREAT_MODEL](THREAT_MODEL.md). Intent: [BRIEF](BRIEF.md).

## 0. Executive summary

- **What.** Train one decoder across machines in different places, of different speeds, joined by wide-area links. The
  result is **byte-identical** to training the same recipe on one device.
- **How.**
  - Pipeline parallelism over contiguous layer slices.
  - **Zero-bubble scheduling** (F / B / W split, Qi et al. 2024) as the only execution model: B (input gradient) is the
    critical path, and W (weight gradient) fills the bubbles 1F1B would leave.
  - The coordinator measures every node and link, profiles F/B/W on each device, cuts layers by measured compute and
    memory, and solves each slot's order with per-stage costs and per-hop communication.
  - Wide stages share a slice across peers in proportion to their throughput (static IWRR, ZeRO-1).
  - Failures heal from spares and checkpoint waves, and the healed run stays bit-identical.
  - **Planned (ADR-0006, M2b): exact sharded stages,** for stages that must be wide (more devices than layer cut
    points) or where width shortens the step (7B on 32 H100 at 400 Gbps, on hosts whose DRAM carries the staged
    copies: 1-GPU hosts at today's transport, 8-GPU hosts only at 2 passes per byte). Peers own per-tensor dim-0 pieces of the fp32 state, keep a working copy in
    the recipe's consumer dtype, stream gradients in their production dtype per module with asynchronous copies, and
    owners fold in microbatch order and update each unit as soon as its fold completes. A per-link stream test, a
    per-unit exposure test and memory admit them; across ~10 Gbps the WAN carries pipeline boundaries.
  - **Planned (ADR-0008, M4): exact Mixture-of-Experts layers.** A router-independent exactness skeleton plus an
    enumerated routing recipe that expresses the MoE layer of Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek-V3/R1, Kimi K2,
    GPT-OSS and Llama 4. Every row set, order and reduction inside the layer is fixed by the recipe, so pipeline cuts,
    zero-bubble, ZeRO-1 and sharded stages carry it; the one addition to their paths is an int64 lane that brings each
    microbatch's expert counts to the owner of a layer's selection bias. Under ADR-0009's HF modules HF's objects and
    names stay and the sparse block, router and experts classes are swapped (Mixtral, Qwen2-MoE and Qwen3-MoE at the
    pinned transformers; the gate's reference carries the same swap); the other four rows serve from-scratch recipes. Expert parallelism is a separate
    decision.
  - **Planned (ADR-0009, M4): any HuggingFace decoder through transformers' own modules.** Each flat-list unit
    (prologue, layer i, epilogue) runs a window of HF's own `ForCausalLM.forward` over one module tree whose parameters
    load by HF names, so every glue op runs where HF runs it; eager attention, a pinned `transformers`, and a launch gate
    that compares the run's unit chain with the vanilla HF model bit for bit on a tiny config of the run's config class
    and refuses any difference. A tied table keeps the stock fold order (a row-sparse pending fold on the Head stage)
    and its rows travel the chain in `TIED` frames.
- **Where it stands (M0).** 160 tests green at M0 (unit, integration, chaos), plus the spec-gate and admission tests added with this document. Every distributed run is diffed bit-for-bit
  against the oracle. On CPU with emulated WAN, the zero-bubble path is 28% faster than the old 1F1B at 40 ms one-way.
  **There is no real-GPU cross-region evidence yet (M1).**
- **Goals.** Byte-exact training; WAN-first scheduling; heterogeneous hardware; recovery from worker loss without a
  wrong bit; no product layer.
- **Non-goals.**
  - Lossy, asynchronous, local-step or order-changing methods: they break the product rule.
  - Marketplace, billing, UI and the Elysium SaaS layer.
  - Tensor parallelism inside a node (later, through an ADR, if the cost model ever demands it).
  - Training through the browser.

## 1. Product

- **Users.** An operator with a fleet of heterogeneous GPUs (and CPUs, or Apple MPS for development) in one or more
  regions, who wants to pretrain or fine-tune a decoder (built-in GPT, Llama-family, a HuggingFace checkpoint, LoRA)
  without a datacenter interconnect, and without giving up exactness.
- **First touch.**
  - `wantrain coordinator --job job.json` on one host.
  - `wantrain worker --coordinator HOST:PORT --advertise-host IP --data-port P [--region R]` on each machine.
  - For one machine: `wantrain local --spec spec.json --compare`.
- **Outputs.** Hex-exact losses, checkpoints, telemetry, and a rationale for every planning decision (selection, chain,
  cut, schedule, precision).

## 2. System model

- **Control plane.**
  - One coordinator; one agent per machine.
  - JSON frames over TCP, admitted by a token HMAC challenge.
  - The coordinator plans, launches, monitors, stores checkpoint waves, and heals. It never touches numerics, with one
    exception: for `arch="hf"` it builds the buffer record once, at the first launch, under a node's environment guard
    and an admitted fingerprint (ADR-0009).
- **Data plane.**
  - One process per slot `(rank, peer)`.
  - Width-1 hops are a chain (rank `r` dials `r+1`, a single duplex connection per hop, `p2p_streams` parallel flows).
  - Wide stages form a full bipartite mesh between adjacent stages, plus an intra-stage clique for ZeRO-1.
  - Frames `FWD`, `BWD`, `END`, `GRADS`, `PARAMS`, `LOSS`, `HELLO`, with lossless codecs.
  - Peers are authenticated by ed25519 (allowlist) or an HMAC token.
- **Processes and protocols.** See CURRENT_ARCHITECTURE §2.
- **Trust boundaries.** See §5 and THREAT_MODEL.md.

## 3. Correctness oracle

**Reference.** `validation/reference.run_baseline` trains the recipe on one device:
- the same seeds (`init_seed`, `data_seed`), batches `make_batch(data_seed, step, mb)`, `M` microbatches per step and
  fold order (microbatch index), and AdamW;
- deterministic kernels (`setup_determinism`: deterministic algorithms, no TF32, cuDNN benchmark off, x86 ISA pin).

**How each mechanism stays byte-identical:**

| Mechanism | Why it cannot change a bit |
|---|---|
| Pipeline cut | The flat layer list composes exactly. The boundary is passed in the recipe's declared dtype. |
| Zero-bubble F/B/W order | It changes *when* ops run, never *what* they compute. The fold is by microbatch index (`OrderedAccumulator`). |
| Deferred dW (`wgrad`) | `dW = dYᵀX` with the same inputs and kernels as autograd. Bit-identical (MODEL-I1), including device threads (MODEL-I2). |
| Runnable-W fill | Moves W earlier in time only. Same fold order. |
| Wide stages / ZeRO-1 | The owner sums all M shard gradients in index order. The update is the same AdamW on the same values. |
| Sharded stages (ADR-0006, planned M2b) | A gather copies the owner's cast bytes: the recipe's autocast dtype for `SplitLinear` parameters when the recipe autocasts, the master dtype otherwise. A single-contribution `SplitLinear` gradient ships as its GEMM's output bytes and widens exactly; the order in which W computes modules changes no bit (each computes `dYᵀX` from its own inputs; a parameter's contributions keep their order). The owner folds its piece in index order, starting from an fp32 copy of microbatch 0's widened piece. Piece boundaries sit on multiples of `lcm(row length, 128)` elements, and AdamW is elementwise; updating unit by unit changes when, not what. Every peer is the reference's device model, also after a heal. |
| Slice-only build (ADR-0007, planned M4) | The recipe carries an init record made once on one host and bound to (config, seed, init program, init code): the generator state before every (phase, layer) of its unchanged init program (the ledger), anchored at the state `manual_seed(seed)` leaves; the sha256 of every initial tensor, taken by the phase-major walk right after the tensor's last write; and the host's sampler fingerprint. A slot runs only the layers that write its tensors, each phase from its recorded state (the same calls on same-shaped tensors from the same state, so the same bytes), checks the state after each phase against the ledger and each of its tensors, once final, against the record, and ends at the ledger's end state, which is the full build's. Under `arch="hf"` the checkpoint overwrites every tensor, but a slot still runs its writers' random phases in a scratch layer and checks each state against the ledger before the load, so the hf post-init RNG state is checked like every other arch's (ADR-0007 R10, closed by the human's decision 6); under ADR-0009 the HF-modules build of a checkpoint draws nothing (parameters on meta, buffers loaded from the run's buffer record), so its ledger is the anchor alone and every slot checks it. A resume loads every tensor and the RNG state from the wave into the model's skeleton, re-tied to its shared parameters, and draws nothing (a user `build(cfg)` model keeps today's full build, then the load). A node whose sampler fingerprint differs from the record's does not generate init and does not heal a slot whose device is the CPU, which computes with its CPU class; deciding that compute class is outside ADR-0007 (R9). |
| MoE layer (ADR-0008, planned M4) | The router runs in the recipe's router dtype (each family's own: the compute dtype for Mixtral, Qwen, OLMoE, GPT-OSS and Llama 4, fp32 for DeepSeek-V3 and Kimi K2) and its score is taken in fp32; the recipe names the score (softmax over E, softmax over the k selected logits, or sigmoid), the selection key (score, logit, or score plus a per-layer selection bias), group-limited selection, renormalization, scaling and where the weight applies; top-k (and top groups) by the stable descending sort (larger key, then lower index); each expert called once per microbatch on exactly that microbatch's routed rows in ascending token index, also on 0 rows, so an unrouted expert's gradient is +0 and AdamW steps every expert; a reduction inside one kernel call on a fixed shape is the kernel's (as for every GEMM), and every sum assembled outside one (renormalization, combine, a token's k dispatch gradients, a tensor's ≥ 3 consumers) is a left fold in a written order; every scatter is a byte move (no atomics); load-balancing and z-loss terms are per layer and per microbatch (or per sequence), their gradients injected at the layer in the oracle and the stages alike; a selection bias is updated after the step from int64 counts (exact in any order), which each microbatch's gradient frames carry to the bias's owner, and is part of the compared state; no RNG. Each microbatch's gradient is then a dense function of that microbatch alone, which is all the fold, deferred W, ZeRO-1 and pieces (for elementwise optimizers) need. The same layer on one CPU device map gives the oracle's bits also in HF's eager form (MEASURED, 40/40); the rules buy dense gradients for wide stages, a selection defined on ties (on identical keys `topk`'s tie sets differ between CPU and MPS on 436 of 20,000 rows), no accumulate-scatter (refused on MPS under deterministic algorithms) and orders the engine does not choose. |
| HF modules (ADR-0009, planned M4) | Every `arch="hf"` unit runs a window of the same `transformers` code the oracle runs whole (the vanilla `ForCausalLM.forward`, pinned version): its input is injected at the unit's entry module and the call stops at the next unit's entry, so rotary tables, masks per layer type, embedding scales, softcaps and multipliers are recomputed per unit from (ids, config) by HF's own code, once, in HF's place and on the unit's device. Every stubbed value (the layers before the unit, as stride-0 views; the prologue on a later unit) is NaN, so a stub value carried to the logits by arithmetic fails the gate; one read through a comparison, `where`, an index or a cast is not caught (ADR-0009 R13, open). The reference is the datacenter's function: on a map of one device class the vanilla `ForCausalLM.forward` in one process, so identity compares the units with HF at full size; a mixed map is refused for every arch for now (the human, 2026-10-11; CONTROL-I25), re-checked at every admission, whose replacement keeps its slot's device class. Rope types whose forward rebuilds `inv_freq` (dynamic; longrope past its original length) are refused. Buffers are bytes of the run: built once, at the first launch, by HF's constructors on the coordinator's CPU under a node's guard and fingerprint, pinned by sha256 per tensor in the JobSpec, stored in every wave, loaded on every node and in the oracle, never rebuilt (CPU math differs by ISA). Attention is the family's own eager function; no forward draws RNG; Linears are `SplitLinear` (bit-identical to `nn.Linear`) except those another op reads outside their forward, which keep dW in B. A launch gate compares the run's unit chain with the vanilla model on a tiny config of the run's config class, at every cut, on the CPU and on the slot's device, bit for bit, checks causality, and refuses any difference (MODEL-I9, MODEL-I10). |
| Tied table across stages (ADR-0009, planned M4) | The stock order, unchanged: per microbatch `g_i = h_i + e_i` (autograd's sum of the head and embedding uses), folded in index order. The dense embedding backward leaves every row outside microbatch i's ids at +0, so the Head stage (owner of the table, its optimizer state and the accumulator) folds those rows as `h_i + 0.0` when `h_i` exists and holds the rest until stage 0's `e_i` rows arrive, resolving microbatches in order: every element gets the oracle's adds in the oracle's order (MEASURED, ADR-0009 Evidence 12). Stage 0 holds the step's rows of the table, runs HF's embedding module and its own backward, and sends `e_i` on its microbatch's ids when its B ends; the Head sends the next step's rows after its AdamW; both travel in `TIED` frames, forwarded fragment by fragment by the stages between (RUNTIME-I19). |
| Recompute | Reruns F from the stored boundary with the same kernels (RUNTIME-I6). |
| Codecs, fragments, multi-stream | Exact byte images (TRANSPORT-I2, TRANSPORT-I4). |
| Adaptive microbatching | Changes fragment size, never tensor values. |
| Heal / restart | Resumes from a sha256-verified wave of every slot at a common step, under the pinned spec (CONTROL-I8, CONTROL-I10). |
| Heterogeneous devices | Bit-comparable to the oracle *on the same device map* (ADR-0005). For now only one-class fleets run: every mixed fleet (more than one device class: device type and model, a CPU's ATen capability) is refused `MIXED-FLEET` at every admission, for every arch, until the heterogeneous track returns (the human, 2026-10-11; CONTROL-I25). Peers of one class may differ in speed. |

What is compared today: the loss at every microbatch (hex), on every run in CI, including across a checkpoint and
resume (RUNTIME-I3). For MoE models (ADR-0008) each layer's aux values per microbatch (hex) are compared too, and the selection biases join the compared state. Weights, optimizer state and checkpoints are **not yet** compared by sha256. A divergence that
leaves every printed loss identical would go unseen. Hashing the final weights and optimizer state of every
integration run against the oracle's is M1 task 3.

## 4. Invariants

| ID | Invariant | Spec ids | Ladder rung |
|---|---|---|---|
| INV-1 | Every mechanism yields losses bit-identical to the oracle | RUNTIME-I1, I2, I7, I18, I19; MODEL-I8, I9, I10; CONTROL-I1, I6, I24, I25 (I5 suspended) | integration, full |
| INV-2 | Every recovery (heal, restart, blip) keeps the run bit-identical | RUNTIME-I3; CONTROL-I8, I10 | chaos |
| INV-3 | Every executed schedule is complete and deadlock-free, for any costs, comm, budget and owner map | SCHEDULER-I3–I7; RUNTIME-I5 | unit |
| INV-4 | The fold order is the microbatch index, independent of arrival | RUNTIME-I4 | unit |
| INV-5 | Deferred dW equals autograd bit for bit, on every backend thread | MODEL-I1–I3 | unit |
| INV-6 | No bytes from the network are executed; payloads are bounded; codecs are exact | TRANSPORT-I1–I4 | unit, static |
| INV-7 | Data-plane peers and control-plane workers are authenticated; there is no default secret | TRANSPORT-I5; CONTROL-I16 | unit |
| INV-8 | One failure convicts one slot; simultaneous losses never convict several; a blip never convicts | CONTROL-I7 | unit |
| INV-9 | The planner and solver either return a valid plan or refuse with the number that refused | SCHEDULER-I8–I12 | unit |
| INV-10 | The solver reproduces the reference heuristic on its own inputs | SCHEDULER-I1, I2 | unit |
| INV-11 | Lint, types, bandit and pip-audit stay clean | — | lint, types, static, security |

## 5. Security model

Trust assumptions (BRIEF): the operator controls the coordinator and admits workers. Workers may be rented machines
on the public internet. Links are untrusted.

| Boundary | Mechanism today | Gap (milestone) |
|---|---|---|
| Worker → coordinator admission | HMAC challenge with a shared token. Without a token the coordinator warns and admits anyone. | Require a token outside local mode (M4) |
| Slot ↔ slot | ed25519 handshake with an allowlist (`trusted_peers`), or HMAC with a token. No default secret. | No encryption (TRANSPORT-I7, M4) |
| Bytes → objects | `serial` (no pickle), `MAX_PAYLOAD`, typed decode | Fuzzing of `serial` and `recv_frame` (M4, `fuzz` rung) |
| Malicious worker (wrong gradients) | `AuditPolicy` exists as a library | Wiring (CONTROL-I13, M4). Until then a malicious worker can corrupt the run undetected. |
| Coordinator compromise or loss | — | Standby (CONTROL-I14, M4) |
| Secrets | Peer key: atomic create, mode 0600. Tokens from env. | — |
| Model code (ADR-0009) | `transformers` pinned with hashes; offline (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `USE_HUB_KERNELS=0` before import); `config.json` is data for the pinned config class, `auto_map` (remote code) refused; any package the pinned transformers probes (`is_*_available`) that is importable refused unless the `hf` extra locks and fingerprints it, and the probe vector equal on every node (an optional import switches the algorithm: mambapy's scan); no `LD_PRELOAD`, `LD_LIBRARY_PATH` or `DYLD_*`; every module's `forward` its class's own; `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; weights only from safetensors by our reader; bytecode compiled from verified sources into a fresh per-launch prefix the worker passes as `-X pycache_prefix` on the stage's command line, an interpreter-start fact in force before any import (`sys.pycache_prefix` is never assigned in-process: assigned after `import torch` it leaves 726/726 torch modules on in-tree bytecode, MEASURED); the guard checks rather than sets, so it may run after `hf.py`'s `import torch`, and asserts every loaded torch, transformers and `wantrain` module's bytecode under the prefix; every file of transformers verified against its RECORD (equal on every node) and of torch and its locked native wheels (`nvidia-*`) against the RECORDs the lock has for the node's platform; every loaded native object (`/proc/self/maps` or the dyld image list) tied to a RECORD file, else recorded by sha256 and equal across nodes of one platform and device class (gpu false: macOS and Linux-CPU arms; CUDA libraries at M1; R18 narrowed); every loaded transformers function's code object equal to its verified source's. Startup only (≈ 3.2 s MEASURED plus 1–4 s of compiling and < 0.5 s of native hashing, HYPOTHESIS): 0% of step time. | A pin bump is a reviewed PR with the census (M4) |

## 6. Failure model

| Failure | Detection | Recovery | Bound on lost work |
|---|---|---|---|
| Worker machine dies | S1 (connection lost past grace) or S2 (stage gone without its done marker) | Heal: a spare restores the common wave and the run resumes | ≤ `ckpt_every` steps + restore time |
| Stage process killed (OOM, container) | S2, at once | Heal | same |
| Stage frozen / data-plane partition | S4 stall while everyone is alive | Restart in place from the last wave | same |
| Control-plane blip | Disconnect inside the grace | Reconnect; no recovery | 0 |
| Several machines at once | ≥ 2 convictions | Run fails with the reason (network, not hardware) | the run, resumable by hand from the waves |
| Engine exception | Non-zero exit with the done marker | Run fails (a replacement would repeat it) | — |
| Straggler (slow but alive) | Profiled at launch; not re-detected mid-run | Cut and ownership by measured throughput at launch. Mid-run re-plan is M2. | throughput only |
| Wide-stage peer loss | S1/S2 | Heal from the checkpoint (M0). Shard replication without rollback is M4. | ≤ `ckpt_every` steps |
| Coordinator loss | — | Run lost after the last wave the operator holds. Standby is M4. | the run |
| Host sleep (dev laptops) | Shows as 900 s stalls | Not an engine failure: keep the host awake | — |

## 7. Workstreams

| Key | Scope | Owned paths |
|---|---|---|
| runtime | Slot execution, JobSpec, oracle, local launch | `src/wantrain/runtime/`, `launch/`, `validation/`, `worker/stage_main.py`, `__init__.py`, bitwise and precision tests |
| model | Layers, precision, data, LoRA/HF, deferred dW | `src/wantrain/model/`, `tests/unit/test_finetune.py` |
| scheduler | Selection, chain, cut, placement, precision, microbatches, zero-bubble | `src/wantrain/scheduler/`, scheduler tests |
| transport | Frames, serial, identity, control channel, shim, codecs | `src/wantrain/transport/`, `src/wantrain/codec/`, transport tests |
| control | Coordinator, agent, liveness, checkpoints, replacement, audit, probes, CLI | `src/wantrain/coordinator/`, `worker/agent.py`, `fault_tolerance/`, `resources/`, `cli.py`, `__main__.py`, control tests, `tests/helpers/`, `examples/` |

Hot files owned by no workstream: `pyproject.toml`, `uv.lock`, `tests/conftest.py`. `tests/chaos/` is owned by the
harness.

<!-- workstreams -->
```json
[{"key": "runtime", "title": "Stage runtime, job spec and oracle", "scope": "Execute one slot's zero-bubble order, fold in microbatch order, AdamW and ZeRO-1, checkpoint and resume, shutdown handshake; JobSpec; the single-device oracle and comparator; local launcher", "ownedPaths": ["src/wantrain/runtime/", "src/wantrain/launch/", "src/wantrain/validation/", "src/wantrain/worker/stage_main.py", "src/wantrain/__init__.py", "tests/integration/test_engine_bitwise.py", "tests/unit/test_user_model_precision.py", "tests/unit/test_runtime_spec.py"], "provides": ["JobSpec.finalize()", "JobSpec.reschedule()", "JobSpec.owned(rank, peer) -> list[int]", "JobSpec.device_for(rank, peer) -> str", "StageRuntime(spec, rank, peer).run()", "OrderedAccumulator(order).add(mb, tensors, verified=True)/release(mb)/reject(mb, replacement)/complete()/result()", "setup_determinism(seed)", "float_hex(x)", "run_baseline(spec, tag)", "compare(baseline_path, candidate_path)", "run_local(spec)", "run_and_compare(spec)", "run_stage(spec_path, rank, peer)"], "consumes": ["model:build_layers", "model:loss_fn", "model:wgrad.capture", "model:wgrad.flush", "model:make_batch", "transport:send_frame", "transport:recv_frame", "transport:Reassembler", "transport:authenticate", "transport:serial.dumps", "transport:serial.loads", "scheduler:iwrr", "scheduler:default_schedule", "scheduler:proportional_partition"]},
 {"key": "model", "title": "Layers, precision, data and deferred weight gradients", "scope": "Deterministic flat layer lists (GPT, Llama-family, HF, user build), precision recipes, fp8 boundary scaling, LoRA, batches and corpora, WeightGradStore-style deferred dW", "ownedPaths": ["src/wantrain/model/", "tests/unit/test_finetune.py"], "provides": ["build_layers(cfg, seed) -> nn.ModuleList", "loss_fn(logits, targets)", "make_config(preset, ...)", "wgrad.capture()", "wgrad.flush(store) -> dict", "wgrad.convert(module)", "wgrad.owned_params(module) -> set", "precision.spec(name)", "precision.classify(compute, boundary)", "make_batch(data_seed, step, mb, mb_size, seq_len, vocab_size, dataset, data_shards)", "lora.apply(layers, rank, alpha, targets, seed)", "hf.load_into(layers, cfg, path, strict)"], "consumes": []},
 {"key": "scheduler", "title": "Selection, routing, cut, precision, microbatches and zero-bubble orders", "scope": "Pure planning functions: node selection and link memory, chain order, min-max layer cut with memory predicate, placement and peer weights, boundary-precision admission, microbatches per step, the zero-bubble solver and its engine bridge", "ownedPaths": ["src/wantrain/scheduler/", "tests/unit/test_zero_bubble.py", "tests/unit/test_schedule.py", "tests/unit/test_scheduler.py"], "provides": ["zero_bubble(cfg, mem_mode) -> Schedule", "simulate(cfg, orders)", "simulate_slots(cfg, slot_orders, mb_owner, peer_weights)", "iwrr(n, weights) -> list[int]", "default_schedule(spec)", "schedule_for_spec(spec, profiles, hops, mem_budgets, mem_mode) -> dict", "solve_microbatches(n_stages, bubble_target, tokens_per_step_cap, mb_size, seq_len, floor, fbw, comm_ratio, cap) -> dict", "assign_layers(n_layers, speeds, floors, ceilings, layer_costs, fits)", "proportional_partition(cfg, speeds, floors, ceilings, costs, fits)", "best_chain(n, edges)", "select_nodes(pool, n, policy, memory, dead)", "place(spec, groups, emulations, keep_partition)", "plan_job(model_cfg, fleet, ...)"], "consumes": ["runtime:JobSpec", "model:GPTConfig", "model:precision.spec", "control:NodeInfo"]},
 {"key": "transport", "title": "Frames, object codec, identity, control channel, emulation, codecs", "scope": "Everything that parses or produces bytes on a socket: frames and fragments, END, payload limit, exact code-free serial codec, ed25519 identity and allowlist, coordinator channel, WAN shim, lossless boundary codecs", "ownedPaths": ["src/wantrain/transport/", "src/wantrain/codec/", "tests/unit/test_transport.py", "tests/unit/test_codecs_identity.py"], "provides": ["send_frame(sock, ftype, step, mb, payload, chunk, codec, shape, frag_idx, frag_cnt)", "recv_frame(sock) -> Frame", "send_end(sock)", "Reassembler().push(frame)", "serial.dumps(obj) -> bytes", "serial.loads(data)", "authenticate(sock, key_path, shared_token, trusted_peers)", "Channel(sock).send/send_file/recv/close", "token_mac(token, nonce)", "LinkSpec", "SenderPacer", "DelayedDelivery", "planes.encode/decode", "planes.encode_scaled_fp8(t, fmt)"], "consumes": ["model:precision.spec", "model:fp8.recover_exp"]},
 {"key": "control", "title": "Coordinator, agents, discovery and fault tolerance", "scope": "Run lifecycle from registration to verified losses: discovery, selection, routing, planning, profiling, schedule solve, launch, liveness, checkpoint waves, heal and restart; node agent; probes, link measurement, profiler, fit; audit policy; CLI", "ownedPaths": ["src/wantrain/coordinator/", "src/wantrain/worker/agent.py", "src/wantrain/fault_tolerance/", "src/wantrain/resources/", "src/wantrain/cli.py", "src/wantrain/__main__.py", "tests/integration/test_coordinator.py", "tests/unit/test_fault_tolerance.py", "tests/unit/test_resources.py", "tests/unit/test_audit.py", "tests/unit/test_control_admission.py", "tests/helpers/", "examples/"], "provides": ["Coordinator(cfg).run_job(job) -> RunResult", "WorkerAgent.run()", "LivenessJudge.judge(states) -> Verdict", "CheckpointStore", "patch_spec_for_resume(spec_json, endpoints, devices)", "find_replacement(spares, dead, survivors, ...)", "AuditPolicy", "NodeInfo", "profile_stage(spec, rank, peer, reps)", "measure_pair(host, port, link, bulk_bytes)"], "consumes": ["scheduler:select_nodes", "scheduler:best_chain", "scheduler:place", "scheduler:plan_job", "scheduler:schedule_for_spec", "scheduler:LinkMemory", "runtime:JobSpec", "runtime:run_stage", "transport:Channel", "transport:token_mac"]}]
```

## 8. Milestones

| Milestone | Artifact | Rungs that must pass |
|---|---|---|
| **M0 — done** (`9f7d81c`) | Engine with zero-bubble as the only path; coordinator; heal; 160 tests | build, syntax, fmt, lint, types, discipline, claims, invariants, unit, integration, chaos, full, static, security |
| **M1 — real-hardware evidence** | `/wan-smoke`; an approved experiment on 2–4 GPUs in two regions with a ~1B model. Measure tokens/s against the same GPUs in one site, predicted vs measured step time, and the profiler calibration. Hash the final weights in every integration run (task 3). | M0 rungs + wan, multinode, bench |
| **M2 — adaptivity** | Runtime telemetry for re-planning (SPEC-runtime 1–2, SPEC-scheduler 2); degrade instead of refuse (SCHEDULER-I13); between-step rebalancing (SCHEDULER-I15). Plan mode for wide stages moved to M2b (ADR-0006) | M0 rungs + bench |
| **M2b — exact sharded stages** (ADR-0006) | Step 1 `wire: native` (production-dtype gradients, plan-ordered per-module flush, asynchronous copies), then step 2 `shard: pieces` (per-tensor pieces, working copies, per-unit updates and gathers without the step barrier, need-first send queues, own-piece and host holds bounded by one step per peer and summed per host, logical checkpoints; `pieces` requires `wire: native`, and a logical checkpoint reads back into `zero1`); admission by summed per-link S1 (host DRAM included, boundary bytes at k passes), per-unit S2 and memory; plan mode for wide stages, with today's `zero1` wide stages and narrower plans kept as candidates, after the engine bridge's lane-order house fix; class-preserving heal (RUNTIME-I10–I13, MODEL-I7, SCHEDULER-I16–I17, CONTROL-I9, TRANSPORT-I8). Local experiments E1–E7 first. | M0 rungs + identity, wan, bench |
| **M3 — ZB-V** | V-shaped two-chunk schedule (SCHEDULER-I14, RUNTIME-I8) | M0 rungs + bench |
| **M4 — hardening and any LLM** | Encrypted data plane (TRANSPORT-I7); audit wiring (CONTROL-I13); coordinator standby (CONTROL-I14); shard replication (CONTROL-I15); slice-only build by ledger replay of the unchanged init program, checked against the recipe's init record, draw-free resume, init on the record's class (ADR-0007: MODEL-I6, RUNTIME-I14, SCHEDULER-I18, CONTROL-I19; local experiments E1–E5 first); serial and frame fuzzing. Exact MoE (ADR-0008, after M2b merges): `MoEMLP` (exactness skeleton, the enumerated routing recipe, per-expert `SplitLinear` experts, shared expert, per-layer aux terms, selection-bias update), the HF-object replacement for the Mixtral, Qwen2-MoE and Qwen3-MoE blocks of transformers 5.19.0 through ADR-0009's table, each block's class swapped to a subclass whose own forward is the exact layer (its adapter arm blocks closeout; the HF-MoE seam rule, ADR-0008 decision 11, lets ADR-0009's trace, classifier and gate accept it; hub checkpoints load through ADR-0009 D7's hub-to-fused map), or, only if ADR-0009 is not granted, Mixtral and Qwen2/3-MoE loading through an allowlist config map (every key mapped, inert or refused; `head_dim`, `norm_eps`, `rope_style`, `qkv_bias`, `qk_norm`) and a fused-layout orientation table, checked on the real config files and against `transformers` 5.19.0 (test-only), with the refusal of unplaced parameters and its `hf_unplaced` rollback, the manifest's `aux` and `moe_bias` sections, planner costs at active and memory and bytes at total parameters, expert exchange units grouped by projection (MODEL-I8, RUNTIME-I18, SCHEDULER-I21, and MODEL-I4 only if ADR-0009 is not granted; local experiments E1–E4, E7 and E2-HF (ADR-0009 granted) or E5 (not granted) first, E6 on M2b's merged code). Any HF decoder (ADR-0009): `transformers==5.19.0` in the `hf` extra; the HF-modules skeleton (parameters on meta, buffers on the CPU), the unit trace and unit windows, strict 1:1 loading and export by HF names, eager attention and the determinism and import guards, the stock-order tied fold (row-sparse, the table on the Head stage, its rows in `TIED` frames on the chain), the buffer record, the launch gate with its causality probe on the CPU and the slot's device, and the fingerprint (transformers' RECORD everywhere, torch's against the lock per platform), the forward-replacement table that ADR-0008's MoE function attaches to; the native hf loader deleted (MODEL-I4, MODEL-I6, MODEL-I9, MODEL-I10, RUNTIME-I19, SCHEDULER-I22, CONTROL-I24; local experiments E1–E8 first) | M0 rungs + fuzz, redteam, identity |

## 9. Performance model

**Per step, for the bottleneck stage with `M` microbatches:**

`T_step = M · (t_F + t_B + t_W)_max + T_bubble(M, costs, comm, budget) + T_opt (+ T_zero1 in wide stages)`

**Per hop and microbatch:**

`t_comm = L + b / B`, where:
- `L` is the one-way latency;
- `b = mb · seq · h · bytes_per_elem` (the boundary, before the codec's lossless ratio);
- `B` is the bandwidth.

The forward activation and the backward gradient travel in opposite directions on a duplex hop. Two conditions decide
everything:

1. **Bandwidth (hard).** The tokens/s ceiling per hop is `B / bytes_per_token`. No schedule moves it. Only a narrower
   *declared* boundary, a better lossless codec, or more bandwidth does. The GPU stays busy if
   `b / B ≤ t_F + t_B + t_W`.
2. **Latency (soft).** The zero-bubble solver hides `L` by running more F in flight. The cost is activation memory
   (with recompute, only the boundary per in-flight F) plus more microbatches per step to amortize warm-up and drain. The
   limit is the largest global batch that still trains well.

**Measured** (M0, CPU, 4 stages, a small model, emulated WAN; one run of 3 reps; s/step):

| One-way latency | Old 1F1B | Old greedy | ZB analytic | ZB profiled |
|---|---|---|---|---|
| 5 ms | 0.634 | 0.711 | 0.662 | 0.732 |
| 20 ms | 1.123 | 0.953 | 0.944 | 0.924 |
| 40 ms | 1.485 | 1.335 | 1.210 | 1.067 |

**HYPOTHESIS** — the engine's own models, falsified by the M1 experiment. Assumptions: bf16 boundary, planes codec
(lossless ratio ≈ 1.43), recompute on, effective 50 TFLOP/s on an RTX 4090 and 400 TFLOP/s on an H100, seq 2048, mb 1,
bubble ≤ 10%, 8 GB activation budget unless stated.

| Model, stages | GPU | F+B+W per microbatch | Bandwidth for ~100% GPU (per direction) | Max hidden one-way latency (global batch) |
|---|---|---|---|---|
| 1.3B, 4 | RTX 4090 | 99 ms | ~480 Mbps | ~800 ms (~1M tokens) |
| 1.3B, 4 | H100 | 12 ms | ~3.8 Gbps | ~100 ms (~1M tokens) |
| 7B, 8 | RTX 4090 | 264 ms | ~360 Mbps | ~800 ms (~1M tokens) |
| 7B, 8 | H100 | 33 ms | ~2.9 Gbps | ~200 ms with a 16 GB budget (~2M tokens); the solver refuses with 8 GB (SCHEDULER-I13) |

**Sharded stages (ADR-0006, DERIVED in `docs/research/adr-0006/derivation.md`, `round3.md`, `round4.md` and `round5.md`; HYPOTHESIS for real hardware).**
Notation: `b_g`/`b_p` are the gradient and working-copy bytes per param (2 for `SplitLinear` params under a bf16 or fp16
recipe, 4 otherwise); `T_mb` is tokens per microbatch; `T_peer` is tokens per peer per step; recompute is on
(8 FLOPs per param per token, so one microbatch of stage s costs `c_s = 8·P^F_s·T_mb/F`, `P^F_s` its GEMM parameters,
of which W and F are `c_s/4` each; the Embed's tables carry none).
- When width pays (`round4.md` §1–§3): cut points are whole layers of `[Embed, Block × n, Head]`, so the Head (`V·h`)
  makes the last stage the slowest (1.65× the others at S 32, 1.08× at S 4) and the Embed's fp32 tables make stage 0 the
  heaviest exchange. A lane of a width-w stage fills and drains at the per-peer op time. At 7B on 32 H100, M 32,
  400 Gbps, the planner takes S 16, w 2 with pieces (966.8 ms per step against 1102.2 at S 32, w 1 and 994.4 with
  `zero1` at the same width) on 1-GPU hosts at today's 10 host passes per byte and on 8-GPU hosts at 2; on 2-GPU hosts
  at 10 it takes `zero1` at that width, on 4- and 8-GPU hosts at 10 it keeps depth (`round5.md` §3); S2 refuses w 4
  and w 8. At 100 Gbps it keeps S 32, w 1. Where width is forced (7B on 128 H100: S 32, w 4) pieces matter most, and
  there stage 0's host DRAM binds first. Memory gives no usable plan at whole-layer cuts: a 1T layer fits at
  w 4 or 8 only at `M/w` that leaves a step of 6–18 C, and a 405B layer fits at w 1. Sub-block cuts are slower at the
  forced-width point once the Head is counted (≥ 52 ms) and load host DRAM alike; they are not built.
- The recipe's left fold cannot be pre-summed, so each microbatch's gradient ships to its owner.
- Links are topological, and a link carries every stage that loads it. Per byte-per-param, a peer's microbatch puts
  `P_s·(w−1)/w` bytes on its NIC and its PCIe path, and `P_s·n_a·n_b/w` on an uplink between sites holding `n_a` and
  `n_b` of the stage's peers. Copies between device and host are asynchronous (≈ 1 ms per step on the compute thread,
  HYPOTHESIS; synchronous copies would cost 419–838 ms at 7B, S 8, w 4). Host DRAM is a link too: it carries each
  exchanged and boundary byte k times per side for every GPU of the host (k = 10 in today's transport, 2 at best when
  staged through the host), so at the forced-width point a 600 GB/s host carries at most 0.64 GPUs of stage 0 (Embed
  + Block 0) today and 3.18 at best, and 1.34 and 6.70 interior GPUs (HYPOTHESIS, `round5.md` §2): today S1 refuses
  stage 0 on every host, and a transport with at most 6 passes (its own ADR) is what admits pieces there.
- **S1, stream, per link:** `Σ_{(s,g) on the link} k·(b_g·P_s·share + A)/c_s ≤ cap`, each stage at its own `c_s`,
  with `A` the pipeline-boundary bytes per microbatch round and k = 1 except on host DRAM (share `2·(w−1)/w` there,
  both directions), summed over the GPUs actually placed on each host. Per NIC at w = 8: H100
  179 Gbps at `T_mb` 4096 and 51 at 16384; RTX 4090 22 and 6.4.
- **S2, exposure, per unit:** the simulation runs each unit's gradient pieces, fold, update and gather as the runtime
  sends them: per-destination send queues served need-first, links shared fairly among the chunks in flight; the plan
  picks backward or forward emission of the last W. `E` less the stage's window must stay within ε of `C`, the
  pipeline's compute per step at its slowest stage. At 7B, S 8, w 4, `T_mb` 4096, 400 Gbps, an interior stage exposes
  18.8 ms (3.6% of C). At the forced-width point (C 871.4 ms, set by the Head stage) stage 0 binds: 24.5 ms (2.8%)
  against 1075.7 ms for today's ZeRO-1, 621.3 for its native-dtype first step and 175.1 for whole tensors (LPT); the
  step falls from 2163.3 to 1124.5 ms where host DRAM admits stage 0 (1-GPU hosts at ≤ 6 passes per byte), and to
  1721.3 ms with the native-dtype first step where it does not. Each owner's fold of M microbatches is device time at every placement (≈ 6.6%
  of C at 7B on 32 H100, HYPOTHESIS).
- Memory per peer k owning `M_k` microbatches: device `b_p + 16/w` B/param of state (6.17 at w 4, 4.17 at w 8, against
  11.0 and 9.5 for today's ZeRO-1), plus its own pieces waiting for their prefix (`b_g·M_k/w`, up to 4 at 7B, S 8, w 4,
  M 32; 32 for the fp32 Embed tables at 7B, S 32, w 4, M 128), one module's dW, in-flight boundaries and the pending
  W; host at most one step of gradients and of gathers, `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w`
  (21.7 GB at 7B, S 8, w 4, M 32, equal weights; 49.2 GB on stage 0 at S 32, w 4, M 128), charged in full and summed
  over the peers of each host.
- Where the predicate refuses pieces, the planner keeps today's `zero1` wide stages and narrower plans as candidates
  and takes the smallest simulated step (7B on 128 H100 at 400 Gbps: `zero1` with the native-dtype first step on
  1-GPU hosts at k 10 and 8-GPU hosts at k 2, S 32, w 1 on 32 GPUs on 8-GPU hosts at k 10 and at 100 Gbps); a job
  that has a plan today keeps one.
- Consequence: across one 10 Gbps link, sharding every stage puts all of them on the uplink (`T_mb ≥ S·40k` for
  H100 at w 2), which device memory refuses; one straddling stage keeps the boundary crossing and adds its exchange. No
  useful 7B placement is admitted on H100 or RTX 4090, so the WAN carries pipeline boundaries, which hide at
  `B ≥ F/(48·h·L_s)` (2–4 Gbps). Cross-site width needs a chained fold (its own ADR).

**Mixture-of-Experts (ADR-0008, DERIVED from public configs; HYPOTHESIS for real hardware).** Dropless routing gives
`Σ_e n_e = T·k` routed rows per MoE layer and microbatch, so compute is that of the k active experts whatever the
skew, while gradient bytes, the working copy and the optimizer state count all E. With `ρ_s = P_s/P^F_s` (total over
active parameters) S1 needs `T_mb ≥ ρ_s` times the dense threshold: at w 4, 400 Gbps, 1,500 dense, 5,521 for
Mixtral-8x7B (ρ 3.68), 9,939 for Qwen1.5-MoE-A2.7B (6.63), 16,431 for Qwen3-30B-A3B (10.95); at 10 Gbps never at a
usable `T_mb`, so across the WAN MoE runs on pipeline boundaries, whose bytes do not depend on E. Device state at
S 8, w 4: 34.8 GB for Mixtral against 9.5 for a dense model of equal active size. The routed part of the pending W is
`T·k·(2h+3f)` values per layer per microbatch (800 MiB for Mixtral at T 4096). A selection-bias update moves
`L_m·E·8` bytes of int64 counts per microbatch per peer on existing frames and waits (184 KB for all of Kimi K2).
Unrouted experts at realistic T: `E·(1−k/E)^T ≈ 0` under uniform routing, a lower bound (a skewed router leaves more). The router runs in each family's own dtype; an fp32 router as an
opt-in costs 0.12% (Mixtral) to 6.85% (Qwen3) of the layer's GEMM work on H100 (HYPOTHESIS). Per-expert calls cost
12E launches per MoE layer per microbatch (7.68 ms against 4.66 ms of GEMM time for Qwen3-30B-A3B at 5 µs,
HYPOTHESIS): fine-grained MoE is launch-bound on GPU, a grouped GEMM is measured at M1, and since the analytic
`c_s` omits launches, S1 and S2 stay conservative for MoE stages. Under ADR-0009's HF modules the experts are HF's 3-D
tensors, whose dW runs in B (DERIVED, T 4096, no recompute): t_B grows ×1.89 / ×1.40 / ×1.66 (Mixtral / Qwen1.5-MoE /
Qwen3-30B-A3B), a zero-bubble schedule keeps 10.6% / 59.8% / 33.6% of its bubble reduction over 1F1B, and the
experts' fp32 gradients raise S1's threshold to 10,882 / 18,981 / 32,357 at 400 Gbps; the routed pending W is 0, but
each bank's fp32 gradient is held from B to W: 5,376 / 1,980 / 2,304 MiB per MoE layer per microbatch (×6.72 / ×7.62
/ ×5.76 the native pending W), times the microbatches between B and W in the zero-bubble order (a 4-layer Mixtral
stage: 22.5 / 45.1 GB at 1 / 2 in flight), charged by the planner in its memory replay.
On that path gate and up run as two f-wide GEMMs (the shapes of the native layer's `SplitLinear`s) where
transformers' own block runs one 2f-wide GEMM, so the replaced block issues 3E more launches per MoE layer per
microbatch than HF's own (4E with recompute): 0.12 / 0.90 / 1.92 ms at 5 µs (HYPOTHESIS), with equal FLOPs.

**HF modules (ADR-0009; MEASURED on the local Mac CPU, HYPOTHESIS for GPUs).** Host time per layer F+B 477 µs against
480 µs for the native block; per unit window a `Model.forward` wrapper (`c_glue`), its mask (`b·S²·4` B written) and
rotary, and one stride-0 NaN-stub call per earlier layer (`c_pass`, no byte written): 2.7–5.5% of an 80-layer model's
last of 8 stages at 400 TFLOP/s, measured per unit by PROFILE. HF's residual sets the boundary: fp32 under fp32
parameters with autocast, so `bytes_per_elem` is 4 (3.40 after the planes codec) and the busy-GPU bandwidths above rise
×2.43 (7B on 8 H100: 7.0 Gbps per direction) until a bf16 working-copy recipe is decided. Eager attention saves
`b·H·S²·s` B per layer for backward, `s` = 6 under autocast with fp32 parameters (fp32 softmax output and its bf16
copy; 4 in pure fp32; +2 for Gemma2's softcap): 3.22 GB at H 32, S 4096, so 8-layer 7B stages with 4 microbatches in
flight cap S near 2.4k on 80 GB, and recompute (one microbatch's layers, 25.8 GB) lifts it. A tied table across stages
adds, on every hop between stage 0 and the Head, `TIED` frames of up to `T_mb·h·4` B per microbatch forward and the
next step's rows backward, and per step `ΔT_tied = M·max(0, (a+ē)/B − max(t_s, a/B)) + Σ_{k<S} max(0, |R_k|·h·4/B −
t_F) + 2(S−1)L + (|T_{M−1}|+|T_0|)·h·4/B + t_AdamW + M·V·h·4/BW_mem + max(0, Σ_{k≥1}|R_k|·h·4/B − W_idle)`
(throughput, warmup, round trip, +0 scan, the backward link's row burst past its idle window; `a` the activation
bytes; `t_s` with the tied head GEMM): Gemma2-2B at S 4, M 8, 10 Gbps, 20 ms, ≈ 197 ms for text and ≈ 414 ms worst
against a ≈ 438 ms step; 0 throughput term when `2a/B ≤ t_s` (DERIVED, HYPOTHESIS; ADR-0009 `round4.md` §1). Per launch: import 1.2–2.3 s, the RECORD fingerprint 3.2 s,
compiling imports 1–4 s, the gate ≈ 1–2 s per device on one core.

**Answers.**
- **Network 10× slower in bandwidth:** throughput falls about 10× once below the busy-GPU line, because the run is
  bandwidth-bound. A 10× higher latency instead costs more microbatches and memory, not throughput, inside the limits
  above.
- **Bandwidth- or latency-bound?** Bandwidth-bound for fast GPUs below their busy line. Latency only matters when
  `t_F` is small relative to `L` (fast GPUs, small models).
- **A worker disappears:** at most `ckpt_every` steps of lost work plus restore time. The result is unchanged.
- **Several stragglers:** absorbed at launch through measured costs (cut, peer weights, schedule). Mid-run drift is not
  re-planned until M2.
- **GPUs of radically different speed:** layers by measured cost and speed, per-stage F/B/W costs in the solve, and
  wide stages weighted by throughput.
- **Communication dominates compute:** use the lossless codecs and a declared narrower boundary (fp8 recipe), more
  parallel streams, or a larger model. Compute grows with `h²` and communication with `h` (SWARM's square-cube
  argument).
- **A malicious worker:** undetected until M4 (audit wiring).
- **A bad scheduler decision:** the runtime W fill absorbs estimate error and jitter. A restart re-solves.
- **The coordinator fails:** the run stops. The waves held by the operator allow a manual resume. Standby is M4.
- **Cost.** `$/1M tokens = Σ_gpus ($/h) / (tokens/s · 3600) · 1e6`. It is measured in M1, and no figure is claimed
  before it.

## 10. Decisions

| Decision | Rationale | Rejected (one line each) |
|---|---|---|
| Synchronous, lossless pipeline | Product rule: a byte-identical result | Async / local SGD / DiLoCo: they change the result. Lossy compression: same. |
| Zero-bubble as the only path | Solved orders hide comm; measured 28% faster than 1F1B at 40 ms | Keeping 1F1B, greedy or interleaved as switches: each doubles the test matrix |
| WeightGradStore deferred dW | W costs only the dW GEMMs | Two-call split: W re-walks the graph |
| Static IWRR ownership | Plannable, and deterministic in timing | Dynamic claim pool: cannot be scheduled ahead |
| Per-stage costs + per-hop comm in the solver | Heterogeneous fleets over WAN | The reference's single median cost: hides the warm-up overrun |
| Profiling on each node before launch | Spec sheets and burns differ from the real F/B/W | Analytic costs only: kept as the default for specs without profiles |
| A coordinator that never touches numerics (one exception: the HF buffer record, built once under a node's guard and fingerprint, ADR-0009) | Simple trust and failure story | Decentralized control: more failure modes, no gain at this scale |
| Chain with one duplex hop, dialed one way | Works through asymmetric NAT | Ring with a wrap link: unprovable reachability |
| No pickle on any socket | Remote code execution | — |
| Exact sharded stages: per-tensor pieces under the recipe's left fold, admitted per link and per unit (ADR-0006) | Bits of the existing oracle at every width. Where width is forced (7B on 128 H100, S 32, w 4, 400 Gbps) the step falls from 2163.3 to 1124.5 ms (1721.3 with the native-dtype first step alone), on hosts whose DRAM carries the staged copies (1-GPU hosts at ≤ 6 host passes per byte; at today's 10 only the first step's 1721.3); device state from 11.0 to 6.17 B/param at w 4 plus the own-piece hold. At 7B on 32 H100 and 400 Gbps width with pieces shortens the step (966.8 vs 1102.2 ms) on 1-GPU hosts today or 8-GPU hosts at 2 passes; at 100 Gbps the planner keeps depth. No usable plan for memory at whole-layer cuts. | Whole tensors (LPT) or flat ranges: units land on few owners (35.3 and 40.5 vs 18.8 ms at an interior stage of 7B, S 8, w 4; 175.1 vs 24.5 ms at the forced-width stage 0). Sub-block cut points: a model change, not built; slower at the forced-width point once the Head is counted. Recipe fold tree or exact accumulators: new oracle. Per-op ZeRO-3: 586 Gbps at T_mb 4096 on H100, and no usable 1T plan either. Chained fold: second fold placement, its own ADR. Stock FSDP: its order varies with the width. |
| Slice-only build by ledger replay of the recipe's init program, checked against its init record (ADR-0007) | The build's host memory falls from `4·P` per slot to one working layer (plus the tied table on its two stages, and the slice itself when the slot's device is the CPU); the resume load (`12·P_s` per slot, unchanged) then bounds the host, and the planner charges the larger of the two. With device homes (HYPOTHESIS, the device split is unmeasured), 70B at S 32 on 8-GPU hosts with 80 GB devices needs 207 GB per host instead of 2207 GB, so it starts and heals on 256 GB hosts; with the home on the host (gpu false) the steady state `16·P_s` binds. Profiling and fit checks hold the slice, not the full model. Init per launch falls from a full walk to the slot's own layers (588 → 94 s at 70B, S 8) after a record made once per recipe (811 s on one core); each resume skips the random init it used to overwrite. No change of the init bits: 84 slices and 168 ledger builds equal the full build, RNG state included (MEASURED). Every first launch proves its initial bytes against digests taken by a different traversal than its own, and every ledger transition, hf included (73 s per 70B hf slot; under ADR-0009 an hf checkpoint build draws nothing and checks the anchor alone); a record of other init code is refused before any draw (ADR-0007 R10, closed). | Do nothing (one file-backed full build per host): 276 GB of disk per host and 879 s per launch. Full-walk replay: 608 s per launch, with its own 708 s digest record. Recipe v2 on torch's samplers: 94 s per launch with digests, as the ledger, and a 710 s cheaper record, for a recipe change that is not portable. A portable counter-based sampler: 46–1745 s per launch by its per-sample cost (HYPOTHESIS), a recipe change and a new native sampler; it is the init-portability decision. Seed checkpoint made once per recipe: 276 GB over the link and 2222 s per launch at 1 Gbps, or 153 s per launch from a per-host NVMe cache of 276 GB. A declared class key (platform, libc, torch version): neither necessary nor sufficient for equal bits. |
| Exact MoE layer: a router-independent skeleton whose every row set, order and reduction is the recipe's, and an enumerated routing recipe (ADR-0008) | The MoE layers of seven named families train byte-identically under every existing mechanism with no new fold or wait (a selection-bias update adds an int64 count lane, `L_m·E·8` bytes per microbatch per peer): a CPU prototype of the softmax row gave 39/39 pipeline arms (fp32, bf16, k 2–4, shared expert, 3 cuts, recompute, an all-tie router) and 16/16 piece arms identical, and 11 of 13 must-fail runs diverged, the 2 others lacking their trigger at that size (MEASURED on 5f2c815; the other rows DERIVED until the local experiment). The planner charges compute at active and bytes and memory at total parameters. Under ADR-0009's HF modules, HF's objects and names stay and each sparse-MoE block of the pinned transformers in its table (Mixtral, Qwen2-MoE, Qwen3-MoE) has its class swapped to a subclass whose own forward is this layer, computing this recipe's bits rather than HF's block's; the other rows serve from-scratch recipes in M4. | HF's block as-is (natively or under an HF-modules adapter): exact between oracle and stage on one CPU map (40/40, MEASURED on a62e4eb), but its `None` gradients for unrouted experts need both guards weakened and an absent marker on wide stages, its `index_add_` is refused on MPS under deterministic algorithms, and its `topk` tie sets differ between CPU and MPS on 436 of 20,000 rows. transformers v5's experts hook: covers the experts only. Dense-masked experts: E/k× compute. Capacity factor: drops tokens, per-rank capacity depends on topology. HF's block tuned (every expert called, a stable sort for `topk`, a gathered add, aux per layer, its 2f-wide gate_up GEMM): exact per device map and allowed, but its fused GEMM gives other bits than the native layer's separate gate and up, so native and HF paths would be two MoE functions; HF-block fidelity and 3E–4E launches per layer are traded for one function. Fused 3-D bank with a grouped GEMM: exact when both paths run the same grouped call, a GPU kernel question for M1 (launch share); only if ADR-0009 is not granted, fused checkpoints load by an orientation table. Global-batch balancing: a bubble up to (S−1)/(M+S−1). Expert parallelism: 4 round trips per MoE layer per microbatch, its own ADR. |
| Any HF decoder through transformers' own modules, windowed per flat-list unit, behind an exact launch gate (ADR-0009) | Fidelity by construction and no family code: 102 of the 178 causal-LM model types of transformers 5.19.0 compute HF's logits and every gradient bit for bit across a cut after every unit, with deferred W (CPU census, MEASURED, `docs/research/adr-0009/`); none differs, and the rest are refused or fail loudly, none silently. Of the 102, 60 are dense decoders this decision delivers; 30 sparse-MoE types wait for ADR-0008 and 12 bidirectional encoders are refused by the gate's causality probe (by class names, HYPOTHESIS until the census re-runs). Stubs are NaN, so a stub value carried to the logits by arithmetic fails the gate (a read through a comparison or `where` is open, ADR-0009 R13). The native path computed other functions under a strict load (max|Δlogit| 8.18e-3 Llama, 0.296 Qwen3, MEASURED). The flat list, planner, profiler, deferred W, pieces and slices keep their contracts. | Fixing the native path per family: coverage linear in families, 17 new causal-LM families per 4–5 months. An own loop with a glue table and a compat shim: consumes the APIs that change every minor (98 of 98 layer signatures between 5.0 and 5.9). HF's forward per stage re-entered through `inputs_embeds` (torchtitan, NeMo): glue applied once per stage (Granite diverges). Two tiers (HF plus native hot families): they win the WAN term (a bf16 cut, ×2.43 fewer bytes: 94 vs 228 ms per hop per microbatch at 7B on 1 Gbps), but a native family computes another function than HF's and costs a port and gate per family; deferred to the precision ADR, whose bf16 HF residual removes the ×2.43 for every family (host time, 480 vs 477 µs, one contended sample, decides nothing). Export and split (torch.distributed.pipelining): exact whole-graph but mask and rotary edges per hop (64 MB at S 4096) or a graph pass, one export per shape. Tied tables: the per-use fold (a recipe change), dense pending tables (45 GB at Gemma3-27B), shipping the summed table per step (≥ 3.8 s at 10 Gbps for Gemma2-2B), a direct Head↔stage-0 link (a topology decision), stage 0 updating the step's rows itself (the Head then waits for them, ≤ 242 ms); a V placement is the planner's once ZB-V lands (`2a` per microbatch per direction, no `ΔT_tied`). |
| Recompute on in production | In-flight depth hides latency at the cost of one F | Stashing graphs: memory caps the in-flight depth |

**Open questions for the human**
1. The target fleet for M1: which GPUs, which regions, and what spend ceiling?
2. The target model scale and the acceptable global batch (this bounds the hidden latency, §9)?
3. Is data-plane encryption required before any run on rented machines (M4 → earlier)?
4. Is coordinator loss acceptable (resume by hand), or is a standby required?
5. MODEL-I6 needs no change of the init stream (ADR-0007). At a first launch only nodes of the init record's sampler
   fingerprint generate init. Decided (the human, 2026-10-09): the init-portability ADR (shipped step-0 slices, or a
   portable sampler with a new recipe version) is not commissioned now; it returns with the heterogeneous fleet.
6. ADR-0008's recipe choices for MoE, decided by autonomy (2026-10-11, each its recommendation): each family's own
   router precision; per-layer or per-sequence aux and z-loss injected locally (`router_aux_loss_coef` never mapped);
   CE, aux hex and selection biases in the identity record; +0 for unrouted experts; M4; loader 6a (no native HF
   loader); the enumerated recipe; the selection-bias update now; the class swap at the pin; hub checkpoints through
   ADR-0009 D7's hub-to-fused map; the HF-MoE seam rule (router and experts rows called as modules, the gate's
   reference carrying the same swap and aux injection, F8 proving the block).
7. ADR-0009's choices, decided by autonomy (2026-10-11, each its recommendation): HF modules for every `arch="hf"` run,
   the native hf loader deleted; `transformers==5.19.0` pinned in the `hf` extra; the fp32 cut in M4 (×2.43 WAN bytes;
   a native tier deferred to the precision ADR), `boundary_dtype` refused for HF; eager attention only; tied tables in
   the stock order (`TIED` frames, `ΔT_tied`); from-config HF runs on the full build; hf LoRA refused; a map no node
   holds whole refused at every admission (R11). Open: NaN-absorbing stub reads (R13), tied tables at width > 1 (R14),
   mixed-map oracle (R16), host-libm constructor floats (R17), OS and driver objects compared but not verified (R18).
   The conflict (CLAUDE.md: the same bits on every hardware mix; §3: per device map) is settled by the human
   (2026-10-11): the oracle holds per device map and only one-class fleets run, every mixed fleet refused for every
   arch at every admission, until the heterogeneous track returns (CONTROL-I25).
