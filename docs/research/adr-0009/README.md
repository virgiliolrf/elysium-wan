# ADR-0009 evidence: scripts, commands, outputs

Derivations behind `docs/design/adr/ADR-0009-any-decoder.md`. Every number the ADR labels MEASURED points here
(script, command, output). Nothing in this directory is product code or a test; no gate reads it.

## Environment (the tree and machine every output ran on)

- Tree: `a62e4eb` (branch `factory/m4`), product `src/` unchanged by this ADR; only `wantrain.model.wgrad` is imported.
- Machine: Apple M4, 16 GB, macOS (Darwin 25.3.0), CPU only, one thread (`torch.set_num_threads(1)`), no network,
  no model download (tiny configs from config classes, parameters from `torch.manual_seed(0)`).
- Python: `.venv/bin/python -I` (3.12), torch 2.14.1; transformers 5.19.0 outside the lock, from the uv-cache
  archives in `scripts/paths.txt` (or any site directory named by `ADR9_SITE`, e.g.
  `uv pip install --target /tmp/adr9-site transformers==5.19.0 safetensors`).
- Run any script as `OMP_NUM_THREADS=1 .venv/bin/python -I docs/research/adr-0009/scripts/<name>.py [args]`.
  `scripts/boot.py` puts transformers and `src/` on `sys.path`; nothing else is imported from the repo.

Provenance. Round-0 scripts were written in the architect's and researchers' session scratchpads; on commit their
absolute paths were replaced by `boot.py` (the only edit, marked in each file's first line). Outputs named
`*-rerun-r1.txt` were produced by these committed copies on 2026-10-11 at `nice 19` while another checkout's pinned
gate held the machine lock: correctness outputs are unaffected; timings are labelled contended. `census-r0.*` is the
round-0 output, kept as recorded. The round-1 census ran the same way (`nice 19`, one thread, outside a test slot:
the slot queue was held by that gate for over an hour; the census is a research script, not a test); see "Census
provenance" below for the runs the orchestrator stopped.

## Evidence map (ADR Evidence numbers)

| ADR item | Script | Output | Result |
|---|---|---|---|
| 1 native vs transformers | `native_vs_hf.py` | `out/native_vs_hf-rerun-r1.txt` | max abs dlogit llama 8.18e-3, llama-yarn 6.26e-3, qwen3 0.296 (argmax 67.2%) |
| 2 stub-and-inject, controls | `stub_inject.py`, `controls.py` | `out/stub_inject-rerun-r1.txt`, `out/controls-rerun-r1.txt` | 70/70 arms exact (gpt_oss refuses sdpa); 1-ulp boundary changes logits; eager != sdpa; sdpa drops Gemma2's softcap; chunked eager changes dK, dV |
| 3 unit windows (round 0) | `proto.py` | `out/proto-rerun-r1.txt` | 22/22 arms exact |
| 4 controls of round 0 | `proto.py` with `CTRL=` | `out/proto-controls-rerun-r1.txt` | `inputs_embeds` diverges for granite only; `to_empty` diverges everywhere; dropping the tied use diverges for gemma2 (tied) only |
| 5 census, round 0 | `census.py` over `proto.py` | `out/census-r0.txt`, `.json` | 86 EXACT, 6 DIFFER, 86 refused/errors/timeouts (178 types) |
| 5 round-0 DIFFERs | `diag_mamba.py` (`MODE=noconvert`) | `out/diag-r1-*.txt` | mamba `dt_proj` loses dW under `convert`, exact without; 2 encoders: a bias aliased under two names split by the harness |
| 5 census, round 1 (final) | `census3.py` over `proto2.py` | `out/census-final.txt`, `.json` | 102 EXACT, 0 DIFFER, 61 refused or errors, 15 not built (> 20M parameters after shrinking) |
| 6 skeleton | `skel.py` | `out/skel-rerun-r1.txt` | no draw; buffers equal a real build; `to_empty` wrong |
| 7 host time | `overhead.py` | `out/overhead-rerun-r1-contended.txt` | native 480 us, HF 477 us per layer F+B (round 0's 858 vs 624 not reproduced) |
| 8 SplitLinear | `split_vs_stock.py` | `out/split_vs_stock-rerun-r1.txt` | y, dX, dW, db equal |
| 9 residual dtype, tied order | `split.py`, `split2.py` | `out/split2-rerun-r1.txt` | autocast: fp32 layer output; bf16 params: bf16; per-use vs stock: 107,254/163,840 differ |
| 10-11 import, churn | `scan.py`, `agg.py` | not re-run (needs four unpacked wheels) | researcher-reported (angle-1) |
| 12 tied row-sparse fold | `tied_rowsparse.py` | `out/tied_rowsparse-r1.txt` | equal to the stock order at n_pend 1, 2, 3, 6; untouched rows +0 |
| 13 threaded state | `ctrl_thread.py` | `out/ctrl_thread-r1.txt` | NaN stubs caught; replay on the gate's ids blind on logits; replay on independent ids refused |
| 14 RECORD fingerprint | `fp_probe.py` | `out/fp_probe-r1.txt` | transformers 0.31 s, torch 2.87 s, all files equal to RECORD |
| 15 export (A5) | `export_probe.py` | `out/export_probe-r1.txt` | 11/11 exact whole-graph at (2, 24); S 40 refused |

Census provenance. Two full in-process runs of `census3.py` (`out/census-r1.txt`, `out/census-r1-part2.txt`) were
stopped (exit 143) by the orchestrator: one process holding every type grew past 5.7 GB and drove the 16 GB Mac into
13.9 GB of swap, with `gemma4_unified`'s unshrunk towers allocating 17-20 GB. The orchestrator changed `census3.py`
to run 10 types per subprocess and to count parameters on the meta device first, recording a type still over 20M
parameters as SKIP instead of building it. `out/census-r1b.txt` is that run, before two harness fixes in
`proto2.py` (the tie found by Parameter identity, not by name prefix; a `None` element in a layer's output passed
through); `out/census-r1b-rerun.txt` re-runs, one process each, the 7 types those fixes or a killed batch touched
(fuyu's batch was killed with SIGKILL; fuyu alone was not re-run). `out/census-final.*` merges the two.
`out/census-final-corpus-check.txt` re-checks Evidence 3's 10 families under the final `proto2.py`.

## What the round-1 prototype changes (`proto2.py` against `proto.py`)

1. Pass stubs return a NaN tensor in place of the residual and NaN tensors for every other tensor element of the
   layer's traced output (traced dims equal to the trace's batch 3 or length mapped to the call's); a non-floating
   tensor element is refused; a non-tensor element is returned as traced.
2. Prologue stubs return NaN (not zeros) for every parameterized prologue module, shaped from its first argument.
3. The trace runs on ids drawn independently of the gate's microbatches, at batch 3 (the gate uses 2).
4. A forward that changes the default generator's state is refused.
5. A `nn.Linear` whose parameter is read by an op outside its own forward (a `TorchFunctionMode` spy over the
   module-call stack) is not converted to `SplitLinear`: its dW stays in B.
6. Parameters aliased under several names are re-tied from the skeleton's sharing map; an alias across units other
   than the (prologue embedding, epilogue head) tie is refused; the tie is found by Parameter identity.

## Round 2 (2026-10-11, tree a446215)

`round2.md` holds the derivations behind the round-2 corrections: side inputs and buffers on a device map (§1), the
unit window's host cost with stride-0 stubs (§2), the tied relay's throughput, warmup and round-trip terms and the
`TIED` frame (§3), the census split by scope (§4: 60 in scope, 30 sparse MoE, 12 encoders; a classification by class
names, not yet a census run), eager attention's saved bytes (§5) and the fingerprint across platforms and bytecode
(§6). `scripts/saved_bytes.py` is new and has not run (the slot queue was held by a pinned gate; E7 runs it first).
No other script or output changed this round.

## Round 4 (2026-10-11)

`round4.md` holds the round-4 derivations: the tied relay's `t_s` with the head GEMM and the backward-link burst term
(§1, `scripts/tied_terms.py` → `out/tied_terms-r4.txt`, arithmetic only), the oracle on a one-class map (§2),
host-libm constants (§3, `scripts/libm_margin.py` → `out/libm_margin-r4.txt`: 1,099 constants, 0 fp32-ambiguous),
rope types that rebuild a buffer in forward (§4) and the init/buffer record order (§5). Both scripts are pure Python
(`.venv/bin/python -I docs/research/adr-0009/scripts/<name>.py`), no torch, no transformers, no test slot.

`round5.md` holds the round-5 text corrections. `round6.md` holds the round-6 correction pass (2026-10-11, autonomy,
no script run): the routing classifier (§1), importable probed packages (§2), A0/A4 against the chosen design on one
cost model (§3), rollback per piece (§4), decision 8 at every admission (§5), the bytecode prefix ordering (§6),
native objects in the fingerprint (§7), the `determinism.py` house item (§8), and the ADR's earlier ideas table (§9,
§10) and Evidence wording (§11), moved verbatim for the 400-line budget.
