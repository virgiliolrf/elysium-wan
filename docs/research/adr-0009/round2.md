# ADR-0009 round 2: derivations behind the corrected cost terms and mechanisms

Each section answers one round-2 finding. Labels: FACT (code read, file:line), MEASURED (command and output in this
tree), HYPOTHESIS (to be measured by the named experiment). Pinned code: transformers 5.19.0 (the uv-cache archive in
`scripts/paths.txt`), torch 2.14.1, tree a446215.

## 1. Side inputs on a device map

FACT: HF computes rotary tables on the device of the tensor it is given (`LlamaRotaryEmbedding.forward` moves
`inv_freq` to `x.device` and runs the matmul, `cos` and `sin` there) and builds masks on the device of
`inputs_embeds`. Under a unit window the tensor is the injected residual, so unit u computes its side inputs on its
own device. A vanilla `ForCausalLM.forward` under an accelerate-style device map computes them once, on the
prologue's device, and moves them.

FACT: no per-map oracle exists today. `validation/reference.py:33-35` runs one device; the mixed CPU+MPS fleet test
(`tests/integration/test_coordinator.py:188-216`, CONTROL-I5) checks that the run trains, not its bits.

Decision (ADR D2): the reference of an `arch="hf"` run on a device map is the unit chain run in one process with each
unit on its map device and the residual copied between devices (a byte copy, autograd's copy backward is a byte copy
too). The run and that reference compute every side input on the same device, with the same code and operands, so
they agree by the D2 argument. Vanilla HF is the gate's comparator on one device at a time, never across a mixed map.
Rejected: computing every side input on the CPU. HF computes them on the residual's device, so this needs a moved
residual or a replaced rotary/mask forward per family (family code, against D1); it removes a difference only on
mixed maps, whose layers already compute other bits per backend (DESIGN §3).

Buffers (adopted from the frontier lens): `inv_freq = 1/(base ** (arange/dim))` runs `pow` on the building host's CPU,
whose vectorized implementation is chosen by the ATen CPU capability (AVX2, AVX-512, NEON). Two nodes of different
ISA can hold other bits of the same buffer (HYPOTHESIS; one Mac cannot show it). The buffers are therefore bytes of
the run: built once by HF's constructors on the coordinator's CPU at the first launch, stored with a sha256 per
tensor in the run record and every wave sidecar, and copied into every skeleton and into the oracle; a node never
uses buffers its own CPU computed. Size: the skeleton's buffers, a few KB to a few MB (Llama 3 8B: `inv_freq`, 64
floats). Rejected: SLEEF's bit-consistent functions or a correctly rounded libm: a second implementation of HF's
constructor math, to be kept equal to HF's per pin, for a problem a few KB of recorded bytes remove.

## 2. Host cost of a unit window

FACT: `scripts/overhead.py:22-23` times the vanilla whole forward, not a unit window; `scripts/proto2.py:94` stubs
with `torch.full_like(hidden_states, NAN)`, which writes the residual's bytes on every stub call.

Per stage and microbatch, with units `u ∈ [a, b)` on the stage and `n_before(u) = u − 1` stubbed layers:

`t_window = Σ_u (c_glue + n_before(u)·(c_pass + w_stub/BW_mem)) + Σ_u b·S²·4/BW_mem`

- `c_glue`: `ForCausalLM.forward` and `Model.forward` around the unit (decorators, output dataclass, mask creation,
  rotary, hooks, the exception exit). HYPOTHESIS 0.2–0.5 ms host, E7 measures it.
- `c_pass`: one stub module call, HYPOTHESIS 3–5 µs host.
- `w_stub`: bytes a stub writes. Round 1's `full_like` writes `T_mb·h·4` (45 MB at h 8192, T_mb 1376). The 80-layer
  model's last of 8 stages runs `Σ n_before = 70+…+80 = 825` stubs per microbatch: 37 GB written, ≈ 18.5 ms at 2 TB/s,
  10.5% of its 176 ms of GEMMs (the reviewer's arithmetic, re-done: 6·8.6e9·1376/4e14 = 177 ms). Decision: a layer
  stub returns `torch.full((), nan, dtype=..., device=...).expand(shape)`, stride 0, so `w_stub = 0`; an in-place
  write into it raises (a loud refusal at the gate, never a silent value; the census counts any family it loses). A
  prologue stub runs once per unit (11 per microbatch here, 0.5 GB, 0.25 ms) and keeps a fresh NaN tensor, because
  `Model.forward` may scale the embedding output in place.
- masks: one `b·S²·4` B mask per unit instead of one per forward. 11 units at S 4096: 704 MB, 0.35 ms at 2 TB/s.

With stride-0 stubs, the example's last stage pays `825·c_pass + 11·c_glue` = 2.5–4.1 ms + 2.2–5.5 ms = 4.7–9.6 ms
of host time per microbatch, 2.7–5.5% of 176 ms (HYPOTHESIS), and twice that under recompute. It is hidden only if
the host runs ahead of the device queue. The planner does not need the model: PROFILE times each unit as it runs
(window included), so the cost is charged as measured. The term grows as `n²/S` per stage and matters most for deep
models cut into many short stages, and for host-bound shapes (small `T_mb·h`).

## 3. Tied table across stages: throughput and latency

FACT: RUNTIME-I19 put `e_i` into "a tied lane of its `FWD` frame for microbatch i" and the next step's rows into
"a tied lane of its `BWD` frames". `e_i` exists only after stage 0's `B_i`, long after `FWD_i` left, and stage 0's
first F of step t+1 needs rows the Head would send only in a step-(t+1) `BWD` frame, which needs that F first: a
deadlock. Decision: a `TIED` frame type. Stage 0 sends `TIED(t, i)` holding `e_i[T_i]` (forward direction) when its
`B_i` ends. After its AdamW of the table the Head sends `TIED(t+1, k)` holding `R_k = T_k \ ∪_{j<k} T_j` (ascending id)
for k = 0, 1, … (backward direction). Every stage between forwards each fragment as it arrives (cut-through) and
never waits for its own compute. Stage 0's `F_k` of step t+1 waits for `R_0..R_k`. The cross-step dependency (Head
AdamW of step t ← `TIED(t, M−1)` ← stage 0's `B_{M−1}` of step t) has no edge back into step t, so it adds no cycle.
A resume sends `TIED(r, k)` from the loaded table before step r.

Terms (`a = T_mb·h·b_cut` activation bytes, `e_i = |T_i|·h·4`, `t_s` the stage's per-microbatch F+B+W, `t_F` its F):

`ΔT_tied = M·max(0, (a + ē)/B − max(t_s, a/B))` (throughput, every hop between stage 0 and the Head)
`        + Σ_{k=1}^{S−1} max(0, |R_k|·h·4/B − t_F)` (warmup: stage 0's first S forwards wait for rows)
`        + 2(S−1)·L + (|T_{M−1}| + |T_0|)·h·4/B + t_AdamW(table)` (the step-boundary round trip, cut-through)
`        + M·V·h·4/BW_mem` on stage 0 (the +0 scan of the dense embedding gradient; the dense backward itself is the
          oracle's and is in `t_s`)

Gemma2-2B (V 256000, h 2304, `T_mb` 4096, fp32 cut and table), S 4, M 8, L 20 ms, B 10 Gbps, `t_s` 30.7 ms
(6·2.0e9·4096/4 FLOP at 400 TFLOP/s), `t_F` 10.2 ms; `a` = 37.7 MB = 30.2 ms:
- worst case (`|T_i| = T_mb`): throughput 8·(60.4 − 30.7) = 238 ms; warmup 3·(30.2 − 10.2) = 60 ms; round trip
  120 + 60 + 8 = 188 ms; scan 8·2.36 GB/2 TB/s = 9 ms. Total ≈ 495 ms against a base step of
  `(M+S−1)·t_s` ≈ 338 ms.
- typical text (HYPOTHESIS: `|T_i|` ≈ 0.45·T_mb distinct ids per 4096 tokens, `|R_k|` ≤ 0.15·T_mb for k ≥ 1):
  throughput 8·(43.8 − 30.7) = 105 ms; warmup 0; round trip 120 + 27 + 8 = 155 ms; scan 9 ms. Total ≈ 269 ms.
- Gemma3-27B (h 5376, V 262144), S 8, M 16: `a` = 88 MB = 70 ms per hop against `t_s` ≈ 207 ms, so `a + e` ≤ 141 ms
  stays under `t_s` and the throughput and warmup terms are 0; the round trip is 280 + 141 + 20 ≈ 441 ms against a
  base step of 23·207 = 4.8 s (9%).

When it disappears: the throughput term is zero once `2a/B ≤ t_s` (large models, fast links); the round trip is the
floor while prologue and epilogue sit on different stages. E4 must make the omitted term observable: on the CPU the
emulated bandwidth is set so that `a/B` equals the measured `t_s` within 20% (tiny gemma2, `h` 256, `T_mb` 512), the
regime in which the throughput term is largest.

V placement (ZB-V, M3's SCHEDULER-I14) puts the prologue and the epilogue on stage 0: `ΔT_tied = 0`, but every link
direction then carries `2a` per microbatch. At the worst case above it costs 8·(60.4 − 30.7) = 238 ms in each
direction against the relay's ≈ 495 ms; at typical text, 238 ms against ≈ 269 ms; when `2a/B ≤ t_s` it costs nothing
and saves the round trip. It is a planner choice once ZB-V exists; the relay is the only option on M4's linear chain.

Row-split update at stage 0 (rejected): stage 0 would fold and update the rows `U_t` itself, but the Head's head
matmul at step t+1 needs the updated `w[U_t]` before its first F: `|U_t|·h·4/B` on the critical path, up to 242 ms at
M 8 (32,768 rows), above the 60 ms bytes term it removes, plus `2·|U_t|·h·4` of moments each step. Early release of
rows (deferred): AdamW is elementwise and today's recipe has no global clipping (grep: no `clip` in DESIGN, specs or
`runtime/spec.py`), so rows outside `T_{M−1}` could be updated before stage 0's last B; it removes at most
`|T_0 \ T_{M−1}|·h·4/B` ≤ 30 ms, while frequent tokens keep `T_0 ∩ T_{M−1}` non-empty, so the round trip stays.

## 4. Census by scope

FACT: `scripts/census3.py` applies neither D8's classifier nor a scope filter. Grep over the pinned modeling files for
classes named `(?i)(sparse|moe$|experts?$)` (this round) splits the 102 EXACT types:

| Column | Count | Types |
|---|---|---|
| In scope, delivered by this ADR | 60 | apertus arcee bitnet bloom cohere cohere2 ctrl cwm diffllama ernie4_5 exaone4 falcon falcon_h1 falcon_mamba gemma gemma2 gemma3_text glm glm4 gpt-sw3 gpt2 gpt_bigcode gpt_neo gpt_neox gpt_neox_japanese granite granite_swa helium hunyuan_v1_dense hyperclovax jais2 lfm2 llama mamba minicpm3 ministral ministral3 mistral modernbert-decoder moshi mpt nanochat olmo olmo2 olmo3 olmo_hybrid persimmon phi phi3 qwen2 qwen3 qwen3_5_text recurrent_gemma rwkv seed_oss smollm3 stablelm starcoder2 vaultgemma youtu |
| Sparse MoE, refused until ADR-0008 (D8) | 30 | aria_text axk1 cohere2_moe deepseek_v3 doge exaone_moe flex_olmo glm4_moe glm4_moe_lite gpt_oss granitemoe granitemoe_swa granitemoeshared hunyuan_v1_moe hy_v3 jamba jetmoe llama4_text mellum mimo_v2_flash minimax minimax_m2 minimax_m3_vl_text mixtral olmoe qwen2_moe qwen3_5_moe_text qwen3_moe qwen3_next solar_open |
| Bidirectional encoders, out of scope | 12 | bert bert-generation camembert data2vec-text electra ernie megatron-bert rembert roberta roberta-prelayernorm xlm-roberta xlm-roberta-xl |

Three of the 30 (Mixtral, Qwen2-MoE, Qwen3-MoE) can train through ADR-0008's SUBSTITUTE rows and then compute that
ADR's function, not HF's block's. The grep is a classification by names (HYPOTHESIS until the census re-runs with
D8's classifier and the causality probe below). It also showed that round 1's classifier `(?i)moe|expert|sparse`
flags non-expert modules of MoE families (`Qwen2MoeRMSNorm`, `Glm4MoeAttention`), which would refuse even a family
with a SUBSTITUTE row; the classifier is narrowed (ADR D8).

Encoders: `BertLMHeadModel` and its kin attend bidirectionally unless `is_decoder` is set, so a causal-LM loss trains
on future tokens while still matching vanilla HF bit for bit. The gate gains a causality probe: two id rows equal
except at the last position must give bitwise equal logits at every earlier position, on the vanilla model; a
failure refuses the config. One tiny forward, milliseconds. For a causal model it holds by construction (masked
probabilities are exactly 0 after the softmax; GEMM order does not depend on values), HYPOTHESIS until E1.

## 5. Eager attention: bytes saved for backward

FACT (`models/llama/modeling_llama.py:194-199`): `attn_weights = matmul(q, kᵀ)·scaling`; `+ attention_mask`;
`softmax(…, dtype=float32).to(query.dtype)`; `dropout(p=0)`; `matmul(attn_weights, v)`.

Derivation per `b·H·S²` element:
- fp32 parameters, no autocast: the softmax saves its fp32 output (4 B); `.to(float32)` returns the same tensor and the
  PV matmul saves it again (same storage). 4 B.
- fp32 parameters under bf16 autocast (today's recipe): `q·kᵀ` is bf16; the fp32 mask promotes the sum to fp32; the
  softmax saves its fp32 output (4 B); `.to(bf16)` makes a copy the PV matmul saves (2 B). 6 B.
- Gemma2's attention softcap adds `tanh`'s saved output in the logits' dtype: +2 B under autocast, +4 B in fp32.

At H 32, S 4096, b 1: 3.22 GB per layer per microbatch under autocast (round 1 said 1.07 GB). A 7B stage of 8
layers with 4 microbatches in flight holds 103 GB, so without recompute the S cap on 80 GB falls by about √3, from
≈ 4k to ≈ 2.4k. With RUNTIME-I6's stage recompute, a stage holds one microbatch's layers at once: 8·3.22 = 25.8 GB
at S 4096 (HYPOTHESIS until E7 measures the saved bytes with `torch.autograd.graph.saved_tensors_hooks` on the CPU).
The probe is `scripts/saved_bytes.py` (llama and gemma2, autocast off and bf16, H 4, S 256 and 512). It has not run:
this round's attempt waited 900 s for a test slot while a pinned gate held the machine-wide lock
(`machine-lock: busy, held by pinned:f1-matrix-x20`, exit 75). E7 runs it first.

## 6. Fingerprint across platforms, and bytecode

FACT: torch ships one wheel per platform and build (macOS arm64, manylinux CUDA builds, CPU-only, ROCm); their RECORD
files differ. transformers is a `py3-none-any` wheel: one RECORD everywhere. Round 1 required equal RECORD hashes of
both on every node, which refuses every mixed-platform fleet (CONTROL-I5's CPU+MPS case included).

Decision: (a) transformers: every file verified against RECORD and the RECORD's sha256 equal on every node; (b) torch:
every file verified against RECORD, and the RECORD's sha256 equal to the entry for (torch version, the node's wheel
platform tag) in `hfm.TORCH_RECORDS`, a table the pin-bump PR generates from the wheels `uv.lock` lists; the
coordinator requires equal torch version and `git_version` on every node and records each slot's torch RECORD hash; a
heal or resume onto another node needs the slot's recorded hash (the device map's bits are per slot build). MODEL-I6's
code id binds (a) plus the torch version and `git_version`, not a per-platform RECORD.

Bytecode (FACT: `scripts/fp_probe.py:17-18` skipped every `.pyc`). uv installs without compiling, and the first
import writes `__pycache__/*.pyc` that RECORD does not list; a timestamp-validated `.pyc` whose header matches the
source's mtime and size is loaded without reading the `.py`. Decision: the launcher sets `PYTHONPYCACHEPREFIX` to a
fresh empty per-launch directory before the first import, so CPython neither reads nor writes in-tree `__pycache__`
and compiles every imported module from the verified source; `__pycache__/*.pyc` is then the only kind of file exempt
from the outside-RECORD rule, and a sourceless `.pyc` elsewhere in either package is refused. The code-object check
recompiles each loaded `transformers` module's verified source and compares every function's code object with the
compiled one by value (`co_code`, constants, names), not by `co_filename`. Cost: compiling the imported modules once
per launch, HYPOTHESIS 1–4 s, measured in E6. Rejected: PEP 552 checked-hash `.pyc` files (the installer would have to
produce them; uv does not by default, and the prefix needs no install change).
