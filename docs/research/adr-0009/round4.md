# ADR-0009 round 4: derivations behind the round-4 corrections

Labels: FACT (code read, file:line), MEASURED (command and output in this tree), DERIVED (arithmetic from stated
inputs, script named), HYPOTHESIS (to be measured by the named experiment). Pinned code: transformers 5.19.0 (the
uv-cache archive `5i-LZiSbH_Aotpp_` of `scripts/paths.txt`), torch 2.14.1. Round 4 changed no mechanism.

## 1. The tied relay's cost, corrected (finding 1)

**`t_s` left out the head GEMM.** Round 2 took `t_s = 6·2.0e9·4096/4` FLOP at 400 TFLOP/s = 30.7 ms. 2.0e9 is
Gemma2-2B's non-embedding count; the tied head adds `2·V·h` FLOP per token forward (`6·V·h` F+B), V·h = 589.8M, i.e.
7.66 layers of 77M. On a balanced partition (33.66 layer-equivalents over 4 stages) `t_s` = 39.8 ms, `t_F` = 13.3 ms,
and the base step `(M+S−1)·t_s` = 438 ms (DERIVED, `scripts/tied_terms.py`). Gemma3-27B was costed from its total
27B (head included), so its `t_s` of 207 ms stands.

**The backward link.** After the table's AdamW the Head sends `R_0` (the round-trip term) and then `R_1..R_{M−1}`
back to back on the backward link, ahead of the step's first `BWD` frame (FIFO, RUNTIME-I19 sets no priority). In the
steady state the backward link cannot be the bottleneck: per microbatch it carries `a + |R_k|·h·4` and the forward
link `a + |T_k|·h·4`, with `R_k ⊆ T_k`, and the throughput term already paces the step by the forward link. Adding a
per-microbatch backward excess on top (the reviewer's `(M−S)·29.7 ≈ 119 ms`) charges one pacing twice. What the
model did omit is the burst: the rows `R_1..R_{M−1}` delay `BWD_0` once they outlast the link's idle window, from the
end of `R_0`'s sending to the Head's first B:

`ΔT_back = max(0, Σ_{k=1}^{M−1} |R_k|·h·4/B − W_idle)`, `W_idle ≈ 2(S−1)·L + S·t_F + (S−1)·a/B + t_B`

(HYPOTHESIS; it ignores the per-hop F of the stages between, so it errs short, toward a larger `ΔT_back`).

Evaluated (`out/tied_terms-r4.txt`, 10 Gbps, 400 TFLOP/s, 2 TB/s, AdamW of the table at 28 B per element):

| Case | throughput | warmup | round trip | scan | back link (burst vs idle) | `ΔT_tied` | base step |
|---|---|---|---|---|---|---|---|
| Gemma2-2B worst, S 4, M 8, L 20 | 165 | 51 | 189 | 9 | 0 (211 vs 290) | 414 | 438 |
| Gemma2-2B text (`|T_i|` 0.45, `|R_k|` 0.15 T_mb), S 4, M 8, L 20 | 32 | 0 | 155 | 9 | 0 (32 vs 290) | 197 | 438 |
| Gemma2-2B worst, S 2, M 8, L 0 | 0 | 4 | 69 | 9 | 75 (211 vs 136) | 157 | 716 |
| Gemma2-2B worst, S 4, M 16, L 0 | 330 | 51 | 69 | 19 | 283 (453 vs 170) | 751 | 756 |
| Gemma3-27B worst, S 8, M 16, L 20 | 0 | 9 | 441 | 45 | 0 (1057 vs 1465) | 495 | 4771 |

So the ADR's 495 / 269 / 338 ms become 414 / 197 / 438 ms; the back-link term is 0 at the ADR's examples and
dominant when L is small and M large. V placement (`2a` per microbatch per direction) costs 8·(60.4 − 39.8) = 165 ms
each way at Gemma2-2B, S 4, M 8 (not 238).

**E4 makes every term observable** (tiny gemma2, h 256, T_mb 512, `a` = 512 KiB; B set so that `a/B` equals the
measured `t_s` within 20%; every arm against an untied baseline with the same V, S, M, B, L and seed):

| Arm | V, ids | S, M, L | Terms it isolates |
|---|---|---|---|
| A1 worst | 8192; microbatch k's ids a permutation of `[k·T_mb, (k+1)·T_mb)` (`|T_k| = |R_k| = T_mb`) | 4, 8, 20 ms | throughput, warmup, round trip; back link 0 |
| A2 burst | as A1 | 3, 16, 0 | back link > 0 |
| A3 small vocabulary | 64, uniform (`|T_i|` = 64, `R_k` = ∅ for k ≥ 1) | 4, 8, 0 | round trip's bytes and AdamW, scan; others ≈ 0 |
| A4 latency | as A3 | 4, 8, 40 ms | the round trip's `2(S−1)L` |

F4 then compares each arm's exposed time with the full formula evaluated from that arm's measured `t_s`, `t_F`,
`t_B`, `|T_i|` and `|R_k|`: a deviation above max(20%, 2 ms) in any arm refutes the model for the term that arm
isolates.

## 2. The identity oracle on a one-class map (findings 2 and 5)

FACT: SPEC-runtime §2 (round 3) made the `arch="hf"` oracle the run's own units in one process. The units are the
mechanism under test, so on that oracle the identity rung compares the distribution of the units, never the units
with HF's function; at full size nothing compares them (the gate runs a shrunk config, R2).

Correction (no new mechanism): the product rule's reference is the recipe trained as a datacenter trains it. On a map
of one device class a datacenter runs vanilla `ForCausalLM.forward`, and that is what `validation/reference.py`
already does for every arch: one device, one `autograd.grad` per microbatch, the index-order fold. With the run's
loaded parameters and recorded buffers it is the oracle there, and the identity rung then tests H1 at full size
(E3). The unit chain in one process stays the reference only on a mixed map one host holds (CPU+MPS), where HF
computes side inputs on the residual's device; there the common mode with the run remains (R15).

## 3. Constants computed on the host by libm (finding 8, part 1)

FACT: constructors compute Python floats that forwards multiply into tensors: `self.scaling = self.head_dim**-0.5`
(`models/llama/modeling_llama.py:215`, CPython `float_pow` calls the C library's `pow`), YaRN's
`0.1·mscale·math.log(scale) + 1.0` (`modeling_rope_utils.py:429-439`), LongRoPE's `math.sqrt(1 + math.log(factor) /
math.log(original_max_position_embeddings))` (`:563-567`), `cos = emb.cos() * self.attention_scaling`
(`modeling_llama.py:113-114`). They are module attributes, not buffers, so the buffer record does not carry them.
`embed_scale` is a buffer (`models/gemma/modeling_gemma.py:57`), MiniCPM3's `residual_scale` uses `sqrt` and `/`,
both correctly rounded by IEEE 754.

HYPOTHESIS (ATen code reading, not re-done here): a Python scalar times an fp32 or bf16 tensor is applied as an fp32
operand, so only its fp32 rounding reaches the bits.

MEASURED (`scripts/libm_margin.py`, `out/libm_margin-r4.txt`): for 1,099 constants (`head_dim**-0.5` for every
head_dim 1–1024; YaRN mscale and its ratio at 15 factors and 3 mscales; LongRoPE at 4 original lengths and 4
targets), every double a libm with error below 1 ULP may return at each call rounds to the same fp32: 0 ambiguous.
So two hosts disagree on these constants' fp32 bits only through a libm off by ≥ 1 ULP or a constant outside the
set. The class stays open for other expressions (R16); closing it is recording each constructor float with the
buffers (a mechanism, follow-up).

## 4. Rope types that rebuild a buffer inside forward (finding 8, parts 2 and 3)

FACT (`modeling_rope_utils.py:47-127`): the rotary forward's wrapper reads `rope_type` per layer type and calls
`dynamic_frequency_update` when `"dynamic" in rope_type`, which recomputes `inv_freq` on the residual's device when
the sequence grows, and `longrope_frequency_update` when `rope_type == "longrope"`, which, when
`max(position_ids) + 1 > original_max_position_embeddings`, computes `long_inv_freq` by `rope_init_fn` on that device
and registers it as `inv_freq`. Both contradict "nothing rebuilds the buffers" and use the device's math. Round 3's
refusal keyed `rope_scaling.rope_type == "dynamic"`, a v4 key; the pinned config holds `rope_parameters`, per layer
type where `layer_types` differ.

Correction (the existing refusal, keyed right): refused, any `rope_parameters` entry whose `rope_type` contains
`"dynamic"`, and a `"longrope"` entry whose `original_max_position_embeddings` is below the recipe's sequence length
(training positions run `0..S−1`). LongRoPE below that length takes the short branch, which re-registers the recorded
`original_inv_freq` buffer (moved, not recomputed).

## 5. The init record and the buffer record (finding 7)

FACT: ADR-0007's init record is made before any launch, on a host the operator chooses (SPEC-control CONTROL-I19);
the buffer record is made at the first launch on the coordinator's CPU (SPEC-control §2.1 step 4). Round 3's MODEL-I6
text had the init record digest every buffer; its digests would then come from the init host's constructors, the
bytes D3 says differ by ISA, and every slot would fail `INIT-DIGEST` against the coordinator's buffers on a mixed
fleet, excluded node by node under CONTROL-I19. Correction: the init record digests the loaded parameters only; the
buffers are checked against the buffer record's pin alone (RUNTIME-I19).
