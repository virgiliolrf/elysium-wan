# ADR-0007 derivations and evidence (research notes; the decision is in docs/design/adr/ADR-0007-slice-only-build.md)

Every MEASURED line below names the file that produced it, its sha256 and the command. All runs: this repository at
tree 318422c plus the uncommitted ADR-0007 documents, macOS arm64 (Apple silicon), torch 2.14.1, Python 3.12.13, ATen
CPU capability `DEFAULT`, `torch.set_num_threads(1)`, run from the repository root with `.venv/bin/python -I <file>`.

| File | sha256 | Round | Exit | Output (verbatim, abridged to the result lines) |
|---|---|---|---|---|
| `probe_replay.py` | `3c6a22b1ce11da300eddc4c5e104b9f130f54eed02c4193a601fecf1485ada53` | 0 | 0 | 63 slices of 3 arms equal; control caught |
| `probe_writeback.py` | `286d28aee849dce8999d311af596f8a43fc64ebc5d45a3449d8fd48fb2af3fd1` | 0 | 0 | `arch_gpt2_untied all contiguous slices equal (weights+rng): True`, `RESULT PASS` |
| `cost_r0.py` | `2b32f3ec4489e54e6d2d240513806e1edc84ad9a1ed9291c195b168b6d05fe75` | 0 | 0 | the round-0 table (70B: 275.9 GB per slot today, `T_init` 588 s) |
| `probe_r1.py` | `a4ec763a9a457e13d6f16d002be78d88a57e3523eadabfc7d30394ec3fdd13ec` | 1 | 0 | section 1 below |
| `cost_r1.py` | `f7d1537324ceb68e5b69209ce7a7b0d87911d36f78f96612ab703a5d2b22642a` | 1 | 0 | sections 4 and 5 below |
| `cost_r2.py` | `09911613104dce1d9e0c8f0788557b22d04f8bb00fcaa07508cba6be8d82bf33` | 2 | 0 | section 7 below (`python3 -I`, no torch) |
| `cost_r3.py` | `0b21cd65ce6b72106804c5b241a2b34e29051114e5f0aac78ee3646a63870050` | 3 | 0 | section 8 below (`python3 -I`, no torch) |
| `probe_r4.py` | `bd183c6fab44b2d424af8fa6be0e0f2b017d7007cadf61e90a9f1de6d8cf67c5` | 4 | 0 | section 9 below |
| `cost_r5.py` | `c265f782c265030b044c60d3bd17afa89b1437e7e11d3eef47a90755b56a9238` | 5 | 0 | section 10 below (`python3 -I`, no torch) |

The round-0 probes are kept verbatim for audit (they insert this checkout's absolute `src` path). Their tied arm bound the non-owned Head to the Embed's table, which
it kept for every slice (`m.out.weight = emb.tok.weight`): the home-write rule of section 2, implemented but not
written into MODEL-I6 in round 0. Round 1's `probe_r1.py` implements that rule generically, from the skeleton's sharing
map, and adds the controls.

## 1. Round-1 measurements (`probe_r1.py`, exit 0, 2.6 s)

```
A builtin: 21 slices equal (owned tensors + RNG end state): True
A general_default_untied: 21 slices equal (owned tensors + RNG end state): True
A general_gpt2_untied: 21 slices equal (owned tensors + RNG end state): True
A general_gpt2_tied_gqa_rope: 21 slices equal (owned tensors + RNG end state): True
A total slices 84 all equal: True
B C1 (tied gpt2 [0,1), non-owned Head write to scratch) caught: True
B C2 (tied gpt2 [L-1,L), owned Head's phase-2 store dropped) caught: True
B C3 (non-owned gpt2 draws skipped, [0,1) untied) caught: True
B C4 (stream stopped at b, [0,2) untied gpt2) caught: True
C ledger (phase-major and layer-major): 168 slice builds of 4 arms equal: True; state size 5056 B
D fingerprint ca47eeb0ec4486ae... 1.61 ms; repeat equal: True
D sha256 2.54 GB/s over 256 MiB (hashlib, 1 thread)
RESULT PASS
```

Three runs gave fingerprint times 1.61, 1.88 and 1.97 ms and sha256 rates 2.30, 2.34 and 2.54 GB/s; the cost model
uses the lowest rate, 2.3 GB/s. Configurations: `n_layer` 4 (6 layers, 21 contiguous slices each). Tied default init
has no arm: `GPTConfig.validate` refuses `tie_embeddings` without `init_scheme="gpt2"` (`gpt.py:124-130`), so the
"Head-only slot of a tied default-init model" case does not exist in this tree.

A probe bug found while writing the ledger arm is itself evidence for E1: a tie target built for real
(`nn.Embedding(V, d)` on the CPU, passed to the Head's constructor) draws `V·d` normals and moves the stream; the ledger's
end state then differed from the full build's. The fix builds it on meta and `to_empty`. E1 keeps this as a control.

## 2. The home-write rule (why tied tables need it, and why it is general)

Let `refs(i)` be the parameters layer `i` references in `skeleton(cfg)`, keyed by object identity (a tied table is one
key referenced by the Embed and the Head). For a slice `[a, b)`, `H = ∪_{i∈[a,b)} refs(i)` is the set of *home* tensors.
The full build's value of a tensor `t` is the last write to `t` in the full build's order. A write to `t ∈ H` can come
from a layer outside `[a, b)`: with `tie_embeddings` and `gpt2`, `gpt2_init` walks the Head after the Embed and
`normal_`s `out.weight`, which is `tok.weight` (`arch.py:191, 211-220`); the last write to the table is the Head's.
Rule: every *writer* of a home tensor (any layer `i` with `refs(i) ∩ H ≠ ∅`), owned or not, runs its phase in a working
layer whose home tensors are loaded from home first and stored back after; tensors outside `H` are scratch. Then for
every `t ∈ H` the sequence of writes the slot applies is the full build's sequence restricted to `t`, in the same
order, each from the same generator state; so the final bytes are equal. C1 and C2 show that dropping either the
non-owned Head's store (Embed-only slot) or the owned Head's store (Head-only slot) changes `tok.weight`.

## 3. The RNG boundary ledger

The generator's state before layer `i`'s phase `p` is a function of the recipe alone (the draws before it). Record it:
`ledger[p][i] = get_rng_state()` (5056 bytes, MEASURED) during one sequential walk, plus `end`, the state after the last
phase. A slot then runs only its writers: for each writer `i`, for each phase `p`, `set_rng_state(ledger[p][i])`, run
the phase, check `get_rng_state() == ledger[p][next(p, i)]` (the next boundary, or `end`), and finally
`set_rng_state(end)`. Exactness: every call a writer makes starts from the state the full build had at that call, and
draws depend only on (state, op, shape, dtype, strides); so each write equals the full build's. Order: within a tensor,
writes keep the full build's order (phase-major: trivially; layer-major: the writers of a tensor are the Embed and the
Head, and the Embed's phases precede the Head's in both orders; a write that reads its tensor, R6, is still ordered).
MEASURED: 168 slice builds (84 phase-major, 84 layer-major) equal the full build, RNG end state included.

Layer-major order lets a slot finish each owned layer in one CPU working layer and store it to its home once, already
trimmed to ADR-0006 pieces: no load-store round trip between phases, and no full-fp32 slice on the device at any time.

The ledger changes no bit and no draw; it moves exact bytes (generator states) instead of recomputing the draws that
produce them. It is not jump-ahead (no GF(2) arithmetic, no private layout: `get_rng_state`/`set_rng_state` are the
public API the checkpoint already uses, `stage.py:721, 756`) and not a draw-count rule (the states are recorded, not
predicted).

## 4. Time and size (`cost_r1.py`, DERIVED from MEASURED per-element costs)

```
7B S8: T_init(full walk) 58 s; ledger slot T/S 7 s + hash 4P_s 1 s = 9 s; record once per recipe 67 s (8 cores); ledger 0.34 MB; working layer 4*Lmax 0.8 GB
70B S8: T_init(full walk) 588 s; ledger slot T/S 73 s + hash 4P_s 15 s = 88 s; record once per recipe 676 s (8 cores); ledger 0.83 MB; working layer 4*Lmax 3.4 GB
405B S16: T_init(full walk) 3463 s; ledger slot T/S 216 s + hash 4P_s 44 s = 261 s; record once per recipe 3984 s (8 cores); ledger 1.29 MB; working layer 4*Lmax 12.8 GB
```

(Superseded in round 2 by section 7.2: the record is one pass.) Record production: pass 1 is one sequential walk that keeps nothing and records the ledger (`T_init`, one core; it
cannot be split, because a state is known only after every draw before it). Pass 2 builds every layer from the ledger,
layer-major, hashes it and drops it; layers are independent given the ledger, so it splits over `c` cores:
`T_rec = T_init + (T_init + 4P/r_sha)/c`. `T/S` assumes balanced slices by parameter count (HYPOTHESIS: the planner
balances by cost, so the largest slice may hold more parameters); the Embed and Head stages of a tied model also draw
the other end's layer (one `V·h` table each).

## 5. Alternatives at one point (`cost_r1.py`): 70B, S 8, w 1, k 8 slots on one 256 GB host, 1 Gbps, NVMe 2 GB/s

| Option | Host DRAM | Local disk | Time to first step (init) | Network | Code that must stay equal to the reference | Mixed CPU classes |
|---|---|---|---|---|---|---|
| Today: full build per slot | 2207 GB | 0 | cannot start | 0 | none | unchecked (silent) |
| Do nothing, strongest: one file-backed full build per host, host lock, slots read their slices | page cache ≤ 256 GB | 276 GB | 864 s (588 + 138 write + 138 read; HYPOTHESIS: sequential I/O; a random-write build under swap can be far slower) | 0 | none (today's `build_layers`) | unchecked |
| Seed checkpoint from a remote generator host | 27.4 GB | 0 | 2884 s (676 generate + 2208 transfer) | 276 GB | the generator | foreign slots admitted |
| Per-tensor interception, full walk | 8.4 GB | 0 | 603 s | 0 | a closed ATen-op allowlist | as replay |
| Replay, full walk per slot (round 0) | 27.4 GB | 0 | 603 s | 0 | `build_slice` orchestration | refused (round-0 key) |
| **Replay with the ledger (round 1, chosen)** | 27.4 GB | 0 | 88 s, after a 676 s record made once per recipe | 0.83 MB | none: one walk program, checked at runtime against the record | filtered at selection; join after the first wave (round 4: device slots only, §9.3) |
| Recipe v2, per-parameter generator | 27.4 GB | 0 | 73 s | 0 | a second recipe version kept forever | unchecked unless the record is added |

The do-nothing option is exact by construction and adds almost no code; it loses on disk (276 GB per host at 70B,
1623 GB at 405B), time (864 vs 88 s per launch, every relaunch before the first wave included) and on the checked
init class, which it lacks.

## 6. Init class: what can be proved, and the bound

Let `f_c` be the function a host of class `c` computes for a sampling call. (a) No mechanism that only schedules, routes
or retries changes `f_c`. (b) A check that a host's bytes equal the reference's needs information about the reference's
bytes at launch: without it, a fleet of one foreign class is indistinguishable from the reference's class, because both
are internally consistent. So the strongest guarantee without shipping weights is: the recipe carries a digest of the
reference's initial bytes (the init record), and every host's bytes are compared with it. A declared tuple (platform,
libc, torch version, capability) is neither necessary (two glibc versions can produce equal bits; the AVX2 `normal_`
path uses ATen's own polynomials, not libm) nor sufficient (one torch version string covers wheels and source builds
with different compilers and FP-contraction flags); it stays a diagnostic. A sampler fingerprint (a fixed probe hashed
in 1.6–2.0 ms, MEASURED) is a necessary condition evaluated before any expensive work, used to filter nodes; the
per-tensor digests are the sufficient check.

Availability: let `n_c` be the nodes in the pool whose fingerprint equals the record's, and `N` the slots a plan needs.
A first launch selects the best fleet among the `n_c` (as `same_region` groups by region) and is refused only when
`n_c < N`, with per-fingerprint counts. Before the first wave a heal needs an in-class spare. After it, init no
longer excludes a node (a resume draws nothing), but that does not make a foreign node's step bits equal: a slot whose
device is the CPU computes forward and backward with the same CPU class whose sampler the fingerprint probes, so a
foreign CPU class also runs other GEMM, `exp` and `log` paths (round 4, section 9.3). This ADR decides init, not the
compute class; for a CPU-device slot the heal keeps the fingerprint as a necessary proxy for it, and only a device slot's
heal ignores it. The cost is zero on a single-class pool and, on a mixed pool, the speed ratio of the best in-class
fleet to the best mixed fleet (HYPOTHESIS until a pool is measured); a foreign node can serve a device slot from the
first wave on, and a CPU-device slot never under this ADR.

## 7. Round 2

### 7.1 Binding and anchor (finding: the record is not bound to its recipe)

FACT: a round-1 record held no seed and no config digest, and a slot took `ledger[p][a]` on trust. A record made with
seed 43 and launched with seed 42 on the same CPU class passes every round-1 check (its chain is a genuine walk, its
digests are the bytes the slots compute) and trains seed-43 weights. Fix, two checks at microsecond cost:
- binding = sha256(canonical JSON of (cfg with `hf_model_sha256` and the LoRA fields, seed, program id, code id)), where
  the program id is the sha256 of `init_phases(cfg)` serialized and the code id the sha256 of the init-path sources
  (`src/wantrain/model/{gpt,arch,lora,hf,wgrad}.py`), `torch.__version__` and `torch.version.git_version` (human
  decision 6, §10.1). `build_slice` compares it before any draw: `INIT-RECORD`.
- anchor: `ledger[first] == get_rng_state()` right after `torch.manual_seed(seed)` (`INIT-LEDGER`).
Claim (proved by induction over the plan): the stages of a plan partition the layers, every slot checks the anchor and
every transition of its writers, and a run starts only when every slot built; so a started run, hf included (its slots
replay their writers' random phases into scratch: §7.6, §10.1), has checked every transition from the seed to `end`. A value-only change of the config (e.g. `init_std`) keeps the ledger and changes the
binding: refused in O(1) instead of after `T_slice` by `INIT-DIGEST`. A signature on the record adds no property: the
record travels inside the job the operator submits, and on every arch every byte of it is either the binding
(checked), a state (checked by some slot) or a digest (checked by the slot that owns the tensor). Round 5 (§10.1) found
this false under hf, where no slot ran a random phase; human decision 6 restores it by the hf scratch replay.

### 7.2 Digests from the phase-major walk (finding: pass 2 checked the replay against itself)

FACT: round 1's pass 2 rebuilt each layer by the slots' own traversal, so a value-only drift common to the replay
(e.g. a working layer whose module names make `gpt2_init` pick the unscaled std: same op, numel and draws) was in the
record and in every slot. Round 2 drops pass 2. The record is one phase-major walk holding one working layer:
- construction runs the real constructor (a tie target is the skeleton tensor `to_empty`, NaN-filled, never built for
  real: a real one draws `V·d`, section 1);
- every later phase runs on the layer's skeleton as it stands before that phase, `to_empty`, NaN-filled;
- after each phase every tensor of its declared write set holds no NaN and every other tensor is still all-NaN, else
  `INIT-PROGRAM`. A phase that reads a tensor propagates NaN into what it writes and is refused, so R6 (an init write that
  reads) is detected at record time instead of assumed absent;
- each tensor is hashed right after its last writer (from the write sets), then the layer is dropped.
The slots run layer-major on the module their constructor built, as `build_layers` does. The two traversals now differ
(fresh NaN skeleton layers vs carried constructed modules), so a drift specific to either fails closed at full scale
(`INIT-DIGEST`); a drift common to both differs from `build_layers` only, and is caught by E1 and by `reference.py`
where the full build fits. The reference also compares `get_rng_state()` at every boundary of its walk with the ledger,
so a scratch-only phase (no digest) is checked too.
Cost (`cost_r2.py`): `T_rec = T_walk + 4P/r_sha` = 70 / 708 / 4168 s (7B / 70B / 405B) on one thread; 58 / 588 / 3463 s
if hashing runs on a second thread (HYPOTHESIS: hashlib releases the GIL on large buffers). Round 1 was 67 / 676 / 3984 s
on 8 cores.

### 7.3 Per-tensor finalize (findings: the tied table hashed and stored at the Embed's last phase)

FACT: `gpt2_init` walks the Head after the Embed and `normal_`s `out.weight`, which is `tok.weight` (`arch.py:191,
211-220`); the round-1 text hashed and stored "an owned layer after its last phase", i.e. the table before the Head's
write. `probe_r1.py` hashes after every writer (line 249), so it never tested that text. Rule: a home tensor is
finalized (hashed, compared, trimmed to pieces, stored to home) once, right after its last writer. A tied table on a
stage holding one end stays on the CPU from its first writer to the Head's last phase: `P_held = V·d_model`, 0.52 /
1.05 / 8.41 GB at 7B / 70B / 405B beyond one working layer of 0.8 / 3.4 / 12.8 GB. E1 adds the control "hash at the
owned layer's last phase" on tied `[0,1)`, which must FAIL.

### 7.4 Where the fingerprint is computed (finding: `probe.measure` dispatches unpinned)

FACT: `probe.measure` (`probe.py:79-133`) imports torch and runs `bench_tflops` without `setup_determinism`
(`determinism.py:21-23` pins `ATEN_CPU_CAPABILITY=avx2` "before the first kernel dispatch"). On an AVX-512 host the
agent process would hash the AVX-512 or libm path while the slot runs AVX2. Fix: the fingerprint and the record are
computed only in a fresh process that calls `setup_determinism` before its first dispatch, in the slot's environment,
and the fingerprint hashes `torch.backends.cpu.get_cpu_capability()` with the probe. Cost: one child process per agent
start (torch import, HYPOTHESIS ≈ 1–2 s), none per step. E5(g) runs the agent unpinned with ATen forced to another
capability and checks the reported fingerprint equals the pinned slot's.

### 7.5 Alternatives with the record as a column (finding: two decisions compared unevenly)

`cost_r2.py` §C, 70B, S 8, k 8 (seconds; "digests" = the slot hashes its slice per launch, the record is made once):

| Option | Per launch, no digests | Once, no digests | Per launch, digests | Once, digests |
|---|---|---|---|---|
| Do nothing, strongest | 864 | 0 | 879 | 708 |
| Full-walk replay | 588 | 0 | 603 | 708 |
| Ledger replay (chosen) | 73 | 588 (the ledger) | 88 | 708 |
| Recipe v2, torch per-parameter generator | 73 | 0 | 88 | 88 |
| Recipe v2, portable sampler, 3 / 20 / 50 / 200 ns per sample (HYPOTHESIS) | 26 / 172 / 431 / 1724 | 0 | 41 / 187 / 446 / 1739 | same as per launch (8 cores) |

With the record on every row: the ledger beats full-walk replay by 515 s per launch and ties v2-torch per launch; v2-torch
saves 620 s once per recipe for a recipe change with no portability gain. The portable sampler's time ranges from 2.1×
faster to 20× slower per launch depending on a per-sample cost nobody has measured here; its value is R1. Code shared by
the ledger and any v2: `skeleton`, writers, per-tensor finalize, digests, draw-free resume, convert by skeleton; specific
to the ledger: pass 1's state capture, the anchor and the per-phase state check. The ledger therefore does not block a
later v2 recipe.

The admission consequence is one rule, not a second policy: a slot that refuses its init (fingerprint, ledger, digest)
is excluded for that record and selection reruns; the HELLO filter is the same rule applied 2 ms before a launch
instead of after one failed launch; the round-1 special heal rule is that rule applied to spares.

### 7.6 hf replays its random phases into scratch (human decision 6)

FACT: `hf.py:347` builds at random and `load_into(strict=True)` overwrites every tensor (MODEL-I4); LoRA is applied last
(`gpt.py:345-348`). The only surviving output of the random phases is the generator state, which the ledger records.
Rounds 2–5 had an hf slot skip the random phases, saving `T_walk·P_writers/P` (73 s per 70B slot), at the price of an
hf post-init RNG state taken on trust from the record (§10.1, ADR R10). Decision 6 gives the saving back: an hf slot
runs its writers' random phases in one reused scratch working layer (≤ `4·P_layer,max`, already budgeted), checks every
transition against the ledger and discards the values; then skeleton slice `to_empty`, strict span load, the LoRA phase
from its ledger entry, then `end`. The induction of §7.1 then holds for hf as for every other arch.

## 8. Round 3 (corrections of text and numbers; the mechanism is unchanged)

All numbers from `cost_r3.py` (DERIVED). Per-element sampler costs and `r_sha` are the MEASURED inputs of section 1;
`r_mem` (5-10 GB/s per pass over fp32 bytes: NaN fill, `isnan` scan with its bool intermediate, trim copy), `c_layer`
(1 ms per (phase, layer)) and `T_skel` (1 s per slot) are HYPOTHESIS until E4 measures them on the same host.

### 8.1 The scheme a config runs (finding: `gpt2` declared on the builtin path)

FACT (read): `_build_arch_layers` sends a config to `arch.build_layers`, the only caller of `gpt2_init`
(`arch.py:232-233`), only when `_uses_general_arch(cfg)` (`gpt.py:368-376`), and that test ignores `init_scheme`
(`gpt.py:433-442`). `validate()` refuses only tie without `gpt2` (`gpt.py:124-130`), so `GPTConfig(init_scheme="gpt2")`
with every other knob default is built today by the builtin path with torch-default draws. Rule (text of MODEL-I6): the
program has a `gpt2_init` phase exactly when the build dispatches to `arch.build_layers` and `init_scheme == "gpt2"`.
E1 gains that arm. The 7B row (bias, learned positions, `d_ff = 4h`, no GQA, untied) is builtin-shaped: it draws torch
defaults, `T_walk` 8.8 s (round 2 priced 58 s). 70B and 405B (no bias, RoPE, GQA) are general: unchanged.

### 8.2 The record walk's memory passes (finding: `T_rec` omitted the fill and scans)

`T_rec = T_walk + 4P/r_sha + (n_fill + n_scan)·4P/r_mem + c_layer·n_phase·(L+2)`, with `n_fill = n_phase − 1` (every
phase after construction starts from a NaN-filled skeleton) and `n_scan = n_phase` (one scan after each phase). Slot:
`T_slice = T_walk·P_writers/P + 4P_s/r_sha + 4P_s/r_mem + T_skel` (the trim copy and the meta skeleton). At 8 GB/s the
omitted terms are 15% of round 2's `T_rec` at every gpt2 scale (0.44 s of 2.9 s at E4's 250M; 104 s of 708 s at 70B),
above F4's 10% bound, as the reviewer computed. Completed: 7B 24 s, 70B 811 s (791-874 s over 10-5 GB/s), 405B 4778 s;
`T_slice` 4 / 94 / 274 s; first-ever launch 70B S 8: 905 s against a 603 s full walk (+302 s); 70B S 32: +244 s.

### 8.3 Host memory: two homes and the resume load (findings: headline in gpu-false mode; resume peak)

FACT (read): `_resume` loads the slot's whole payload with `torch.load(path, map_location="cpu")` (`stage.py:747`):
the chunks' fp32 state, AdamW's `m` and `v` (`stage.py:715-718`), `12·P_s`, plus the slice itself when its home is the
host. Today's init peak `4·P` exceeds it whenever `3·P_s,max < P` (`S > 3` with balanced slices; equal at `S = 3`;
round 5, `cost_r5.py` §C), so MEM-INIT covered the resume without saying so. With the
slice build the resume becomes the binding host term:

| Plan, k 8 | today per host | device home: build / with resume | home on host (gpu false): build / with resume and steady |
|---|---|---|---|
| 7B, S 8 | 215 GB | 6.4 / 80.7 GB | 33.3 / 107.6 GB |
| 70B, S 8 | 2207 GB | 27.4 / 828 GB | 303 / 1104 GB |
| 70B, S 32 | 2207 GB | 27.4 / 207 GB | 96.4 / 276 GB |
| 405B, S 16 | 12987 GB | 102 / 2435 GB | 914 / 3247 GB |

70B at S 8 trains nowhere (`16·P_s` = 138 GB per device or per gpu-false slot); it is an init-only comparison point. 70B
at S 32 on 80 GB devices (34.5 GB per device at `16·P_s`) needs 207 GB per 8-slot host with this ADR and 2207 GB today:
startable and healable on a 256 GB host (HYPOTHESIS: device split unmeasured, `B_0` excluded). With the home on the host
the steady state `16·P_s` binds: 276 GB at k 8, so k ≤ 7 on a 256 GB host. A per-tensor resume load (`torch.load(...,
mmap=True)`) would remove the resume term; it is a new mechanism, recorded as a follow-up (ADR R7).

### 8.4 E5(g) on a single-capability host (finding: vacuous green)

FACT: this host reports ATen capability `DEFAULT` and `determinism.py:21-23` pins only x86, so a pinned and an unpinned
process compute the same fingerprint here. E5(g) splits: (g1) structural, runs here: HELLO's fingerprint must come from a
child process whose entry calls `setup_determinism` before importing any module that dispatches a kernel (at the call
the child's `sys.modules` holds no `probe`); its WOULD-FAIL-IF is a HELLO computed in the agent process. (g2) behavioural: N/A where `get_cpu_capability()` has one value,
recorded pending; runs on an x86 host or amd64 container reporting `avx2` or `avx512`, where the unpinned regression
must fail.

## 9. Round 4 (corrections of text and amendments; the mechanism is unchanged)

### 9.1 `to_empty` unties shared parameters (finding: the S 1 tied resume)

MEASURED (`probe_r4.py`, exit 0, verbatim): `meta: tied True params 1`, `to_empty: tied False params 2`, `re-tied: tied
True params 1`, `load strict ok; head bytes equal ref True still tied True`. So `Module.to_empty` on torch 2.14.1 gives
each module that referenced a shared meta Parameter its own new Parameter. A slice that holds both ends of a tied table
(S 1, or a single wide stage) would gain one trainable tensor: `opt.load_state_dict` refuses the wave's group size, and
a rebuilt optimizer would train two copies (wrong bits). The skeleton already carries the sharing (MODEL-I6: "names,
shapes, sharing"); the round-3 text assumed `to_empty` keeps it. Correction of that premise, not a new mechanism: every
materialization of a skeleton slice (resume, hf build, PROFILE, FIT) re-ties from the sharing map the skeleton gives
before `to_empty` (canonical name to its first name, `named_parameters(remove_duplicate=False)`), before any load, and
checks one Parameter per shared group. Cost: O(names) once per materialization, µs; 0% of `T_step`. A Head-only stage
holds one end only: nothing to re-tie.

### 9.2 Non-persistent state and `arch="user"` (finding: the user-model resume)

FACT (read and MEASURED): `grep register_buffer src/` finds nothing; `rope_cache` is computed in `forward`
(`arch.py:73-77`); `probe_r4.py` prints `buffers in builtin 0` and `buffers in general rope swiglu rmsnorm gqa tied gpt2
0`. So for builtin, general and hf every tensor a layer holds is a Parameter in `state_dict()`, and skeleton `to_empty`
plus a strict load restores all of it. A user `build(cfg)` may register a non-persistent buffer (absent from the wave:
`to_empty` leaves it uninitialized and the strict load cannot see it), keep a plain tensor attribute (left on meta), or
be unbuildable on the meta device. Correction: `skeleton(cfg)` is defined for builtin, general and hf only; `arch="user"`
keeps today's resume, PROFILE and FIT: the full build, then the load and `set_rng_state` from the wave (the build's
draws are overwritten). The refusal of a wave without the RNG state applies to it unchanged.

### 9.3 Compute class of CPU-device slots (finding: E5(e) overclaimed)

FACT: with gpu false every slot computes on its CPU. `determinism.py:17-24` pins x86 to AVX2 and leaves ARM unpinned,
so an arm64 spare and an x86 reference differ in GEMM and transcendental kernels as well as in the sampler. Skipping
the draws (the resume) does not make their step bits equal. `find_replacement` (`replacement.py:20-50`) matches device
model, region and dtype, never the CPU class. Correction within this decision's mechanism: for a run that carries a
record, a CPU-device slot's replacement must have the record's fingerprint (the same comparison the exclusion rule
already makes, applied at that heal), a necessary and not sufficient proxy; a compute-class check is R9 (open). E5(e)
asserts the exclusion for CPU-device slots; a device slot taking a foreign-fingerprint spare is N/A while gpu is false.

### 9.4 Tied hf Head-only stage (finding: `out.weight` never written)

FACT (read): the only write to a tied hf output table is the load of `model.embed_tokens.weight` into the Embed's
`tok.weight` (`hf.py:284`); `load_into` drops a tied `out.weight` from `missing` (`hf.py:315`). Under the writers rule a
home tensor's writers include the load of its checkpoint source, even when that source belongs to a layer outside `[a,
b)`. Correction: a stage holding the tied Head but not the Embed reads that one checkpoint tensor (`safe_open` /
`get_tensor`, already used at `hf.py:196-198`) into its `out.weight`; the span load counts the tied `out.weight` as
required on such a stage; every hf home tensor is finalized against the record's digest, as on the other paths. Host
cost: `V·h` fp32 transient, already in `P_held` (1.05 GB at 70B).

### 9.5 PROFILE and FIT (finding: the headline needed them and nothing checked them)

FACT (read): `profile.py:55` and `fit.py:55` run `build_layers` (`4·P`) in each profiling subprocess before launch;
`fit.py` returns `skipped` on a non-CUDA device unless forced. CONTROL-I19 already moved them to the skeleton (round 1);
the ADR text now states it, with the bound: a profiling subprocess holds its slice (`4·P_s`), its gradients and
activations, never `4·P`. Falsifier (E3): at the 250M point, a PROFILE subprocess, and FIT with `force=True`, must save
at least 90% of `4·(P − P_s)` against today's full-build PROFILE or FIT of the same stage (0.75 GB at S 4), and a
structural check finds no call of `build_layers` in `profile.py` or `fit.py` outside `build: "full"` and
`arch="user"`. The fill is from a private generator seeded 0: memory does not depend on the values. Round 5 (§10.6):
CPU timing may (subnormal operands), so the claim on timing is withdrawn and E4 measures it.


## 10. Round 5 (corrections of text, numbers and amendments; the mechanism is unchanged)

All numbers from `cost_r5.py` (DERIVED; inputs as section 8; it leaves `cost_r3.py` unchanged).

### 10.1 hf: the random phases' states are trusted, not checked (two majors, one defect)

FACT (ADR text and `hf.py:343-347`): under hf a slot runs no random phase (§7.6). The anchor check links
`ledger[first]` to `manual_seed(seed)`; with LoRA, each slot checks its writers' LoRA transitions, so the chain from
`ledger[lora][first]` to `end` is checked; nothing links `ledger[first]` to `ledger[lora][first]` (to `end` without
LoRA). The LoRA adapters' bytes come from their own `(seed, index, name)` generator, so the digests do not see the
global state either. Consequence: a record whose post-anchor random-phase states or `end` are wrong but internally
consistent passes `INIT-RECORD`, `INIT-LEDGER` and `INIT-DIGEST`, and `set_rng_state(end)` installs a post-init RNG
state that the identity manifest compares (`state.py:91`): one differing field against `FACTORY_REFERENCE=1`.
Non-adversarial path: the program id hashes `init_phases(cfg)` (phase names and write sets), not the code that draws.
A record made by checkout A and launched on checkout B whose construction draws differently (e.g. `arch.Block`
creating `proj` before `q`, `arch.py:121-131`) keeps the binding and the loaded bytes; nothing raises. For non-hf arches
the same drift is caught, late, by `INIT-LEDGER` after `T_slice`, because the slots replay every transition.
Correction (text): the claims of §7.1, the ADR's security paragraph, F5 and E1's "perturbed ledger entry" control are
scoped to non-hf runs. For hf they state the trust, and E1 pins it as a strict expected failure (a perturbed hf
random-phase entry, or `end` without LoRA, is NOT refused) citing the finding opened for R10.
Options for the human, none adopted (each is a new mechanism; ADR R10, question 6):
- (a) bind the record to the code that draws: add to the binding the sha256 of the init-path sources (`gpt.py`,
  `arch.py`, `lora.py`, `hf.py`, `wgrad.py`) and `torch.__version__` plus build hash. O(1), µs per launch, 0% of
  `T_step`; closes the non-adversarial drift for every arch and turns non-hf drift into an early `INIT-RECORD`; does not
  close a deliberately forged record (the operator's trust, as the record travels in the job).
- (b) each hf slot replays its writers' random phases into one reused scratch buffer of the largest tensor's size
  (same op, numel, dtype and strides; values discarded): the anchor-plus-partition induction then holds for hf. Cost:
  `T_walk·P_writers/P`, 73 s per 70B slot (the saving of §7.6 given back), scratch ≤ `4·P_layer,max` already budgeted.
- (c) accept: the hf post-init RNG state is trusted from the bound record, checked where `reference.py` runs.

Human decision 6 (2026-10-09): (a) and (b) both. With (b) every transition, hf included, is checked by the slot that
owns its layer, so a wrong but consistent record fails closed (`INIT-LEDGER`); with (a) a record from other init code or
another torch is `INIT-RECORD` in O(1) on every arch. The scoping of §7.1, the security paragraph, F5 and E1 to non-hf
runs and E1's hf expected-failure pin above are withdrawn: E1's perturbed-ledger control covers every arch, and a record
of another code id is a control that must fail. No finding was opened for R10; R10 is closed.

### 10.2 The exclusion rule's scope (minor)

`INIT-RECORD` (binding) and an `INIT-LEDGER` at the anchor are properties of the record against the job and are equal on
every node; excluding nodes for them reruns selection until the in-class pool is exhausted (32 nodes, N 8: four
launches, then a capacity refusal that names the wrong cause). Correction: such a report refuses the job once, naming
the record; the exclusion rule applies to `INIT-CLASS`, an `INIT-LEDGER` at a transition and `INIT-DIGEST`. Checking the
binding and anchor at admission in the control plane, with no node, is a follow-up (a new placement of the check).

### 10.3 The per-layer LoRA phase (minor)

FACT: `lora.apply` raises when no Linear of its list matches (`lora.py:113-119`); a per-layer call on the Embed, or the
Head under `attn`, would raise. Correction: the per-layer LoRA phase is the adapter-wrapping primitive (freeze, wrap
the matching Linears, seed from the global index) without the empty-match error; the "matched no Linear" refusal and
`WT_LORA`'s trainable and frozen counts stay whole-program checks, computed from `skeleton(cfg)`. E1 gains a LoRA arm
whose slices include an Embed-only and a Head-only stage.

### 10.4 The record's encoding and size (minor)

The encoding is fixed: JSON with each 5056-byte state base64-encoded (6744 characters) and each digest as 64 hex
characters under its canonical name. `cost_r5.py` §B: 70B gpt2 1.23 MB (ledger 1.12, digests 0.11); 70B with LoRA
`all` 1.95 MB; 405B gpt2 1.92 MB; 405B with LoRA `all` 3.05 MB. The constant 4 MiB cap is replaced by
`cap(cfg) = (n_phase·(L+2) + 1)·6760 + Σ_t (len(name_t) + 80) + 65536` bytes over `skeleton(cfg)`'s names, checked
before parsing: a valid record always fits, and an oversized one is refused at the boundary.

### 10.5 Alternatives, each charged its own record (minor)

`cost_r5.py` §A (70B, S 8, k 8; seconds; with digests): do nothing 879 per launch, 708 once (a walk that hashes the
full build; no ledger, no NaN passes); full-walk replay or interception 608 / 708; seed checkpoint generated once and
shipped per launch 2222 / 708, 276 GB over the link; seed checkpoint cached on the host's NVMe 153 per launch, 708 once
plus one 2208 s transfer per host, 276 GB of disk per host; ledger 94 / 811. Without digests the per-launch figures fall
by 15 s; the other options then pay nothing once per recipe, and the ledger still pays its 588 s walk. Ranking
unchanged: 94 s against 153 s (the closest, with 276 GB of disk per host and a per-host transfer).

### 10.6 Rates from one run; PROFILE's fill (minors)

`t_u` and `t_n` are one run at 256 MiB; every 70B and 405B time scales with them, and F4 tests the model's additivity at
250M, not those headlines. E4 re-measures `t_u` and `t_n` over 5 runs at 2^28 elements (1.07 GB, at least the largest
70B tensor, the 32000 × 8192 table) and the ADR labels the 70B and 405B times HYPOTHESIS (extrapolated) until then.
PROFILE and FIT: memory does not depend on the fill; CPU time may (subnormal operands, x86 without FTZ/DAZ). The
partition changes no bit, so this is cost-model accuracy only: E4 profiles one stage with the private fill and with the
recipe's built values, 5 runs each; a median gap above 10% reopens the fill rule (a follow-up).

### 10.7 The identity manifest under `arch="user"` (minor)

`validation/state.distributed_manifest` takes names, shapes and sharing from `skeleton(cfg)` for builtin, general and
hf, and from today's full build for `arch="user"`, which has no skeleton (SPEC-runtime task 4, amended).

## 11. Ideas considered and changelog of rounds 0–4 (moved verbatim from the ADR in round 5, to keep it within 400 lines)

| Idea (source) | Verdict | Reason and number |
|---|---|---|
| Streamed replay; writers rule; RNG boundary ledger; trim at the final store; skeleton for PROFILE and FIT; draw-free resume, RNG required; per-tensor digests; behavioral fingerprint; one phase list (rounds 0–1); bound and anchored record; digests from the phase-major NaN-filled walk; per-tensor finalize; pinned fingerprint process; reference checks the ledger; hf replays nothing random; record as a column (round 2) | ADOPTED | 84 slices and 168 ledger builds exact (MEASURED); 588 → 94 s per 70B launch; the only sufficient check, 15 s per 70B launch; fingerprint 1.6–2.0 ms; closes the wrong-seed hole and the circular check; +1.05 GB held at 70B on two stages; −73 s per 70B hf slot |
| One walker per host, mmap; torchdistx; shipped step-0 slices; Megatron CPU init and interception; CORE-MATH (rounds 0–1); count-only pass with jump-ahead; signed or Merkle record; record-derived HELLO probe (round 2) | REJECTED / FOLLOW-UP | Same core-time (8 × 73 ≈ 588 s), 276 GB disk; archived dependency; 276 s per foreign 70B slot at 1 Gbps; interception gains only past R2; digests need the values; every byte is re-checked; HELLO precedes any job |
| Portable counter-based sampler (round 2); JAX partitionable threefry (rounds 3–4: reported the default since JAX v0.5.0, not re-checked here); a portable sampler's own correctly rounded `logf`/`sincosf` or a transcendental-free transform, as glibc 2.41 does not round those correctly (round 4, frontier) | DEFERRED to the init-portability ADR | 46–1745 s per 70B launch by per-sample cost (HYPOTHESIS); a recipe change; its value is R1; the libm point narrows that ADR's design space, not this decision |
| BLAKE3 or per-piece digests; hashing on a second thread; step-0 wave for the pre-wave heal (round 2); per-tensor resume load, `torch.load(mmap=True)` (round 3, R7); a `resume` field rolled back alone (round 3, R8) | FOLLOW-UP | ≤ 15 s per 70B launch; `T_rec` 811 → ≈ 691 s (HYPOTHESIS); a shipped wave is 2208 s at 1 Gbps; resume host term 828 → ≈ 27 GB per 70B S 8 host; a runtime rollback at 0% of `T_step` |
| Static order check of multi-writer tensors; `TorchDispatchMode` write and read sets; record emitted by the reference build; binding to program and seed only; binary ledger sidecar; jump-ahead check of hf's skipped phases; quorum attribution of `INIT-*`; fingerprint inside `fork_rng`; recipe-derived probe values (round 3) | FOLLOW-UP | Only the tied table has two writers; −104 s of 811 s at 70B; removes `T_rec` only where a reference run exists; ledger 0.83 / 1.29 MB under the 4 MiB cap; ms per slot; one refusal instead of up to O(pool) relaunches; µs each |
| Skeleton completeness check (every buffer persistent, every tensor attribute a Parameter or buffer, else full build) (round 4, alternatives-and-risk) | FOLLOW-UP | O(modules) once per launch; today no builtin, general or hf module holds a buffer (MEASURED) and `arch="user"` keeps the full build, so it guards only future arches |
| Seed checkpoint cached per host on NVMe, priced as a baseline (round 4, alternatives-and-risk) | FOLLOW-UP, for the init-portability ADR | ≈ 138 s per 70B S 8 launch at 2 GB/s NVMe (HYPOTHESIS), sampler-independent; its value is R1, not this decision |
| Fingerprint as a preference key in `find_replacement`; a compute-class fingerprint (GEMM, softmax, GELU probe) for CPU-device slots (round 4, alternatives-and-risk, first-principles) | FOLLOW-UP (R9) | 0 per step, ≈ 2 ms per HELLO; this round makes the record's fingerprint required for a CPU-device slot's heal, which subsumes the preference; the compute-class probe is a new mechanism |
| Re-tie by `torch.__future__.set_swap_module_params_on_conversion(True)` or Accelerate-style `retie_parameters`; read hf `embed_tokens` by `safe_open(...).get_tensor` (round 4, frontier, consistency) | FOLLOW-UP (implementation choice) | The round-4 correction requires one Parameter per shared group and the `embed_tokens` read; how is the implementer's, µs per materialization, `V·h` already in `P_held` |
| HELLO's fingerprint marked declared under CONTROL-I11 (round 4, consistency) | FOLLOW-UP | Text only; the digests make a lying worker harmless to the bits |
| Zero-copy hashing through the buffer protocol (round 4, first-principles) | FOLLOW-UP | `.tobytes()` adds a transient of `4·` the largest tensor (1.05 GB at 70B, 131 MB at E3's 250M); E3 must hash without the copy or its bound reads high |

- Rounds 0–3: premise refuted, v2 rejected (0); writers rule, record, ledger replay (1, 10 majors); bound record,
  NaN-filled digests, per-tensor finalize, pinned fingerprint, one exclusion rule (2, 8); text and numbers only (3, 8).
- Round 4 (2026-10-09): 8 majors (two user-resume ones the same defect; the relay truncated the eighth, confirmed by
  `probe_r4.py`), all valid, no mechanism added: `to_empty` unties, so materialization re-ties from the skeleton's sharing
  map; `arch="user"` keeps today's resume, PROFILE and FIT; a tied hf Head-only stage reads `embed_tokens`; PROFILE and
  FIT in the proposal, model, F3, E3; E5(e), Q2: CPU-device heals keep the fingerprint (R9); SPEC-model §2 and task 1
  cut to this ADR; derivation §6, §9. Eleven ideas: follow-ups.
