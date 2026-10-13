"""The schedule solvers: bubble arithmetic, lifespans, rank ordering. Zero
bubble (Qi et al. 2023) splits the backward into B and W and this runtime banks
W passes to spend inside stalls; controllable memory (Qi et al. 2024) reduces
peak activation memory to lifespans, so rank 0 is the hungriest slot."""

import pytest

from wantrain.scheduler.schedule import (
    fbw_costs,
    solve_k_inflight,
    solve_microbatches,
    stage_live_graphs,
)

# CONTROL-I25: every fleet here is one device class (speeds may differ).
ONE_CLASS = "cpu:x86_64:AVX2"

# ---------------- fbw_costs: the Table-1 identities ----------------


def test_fbw_identities():
    tf, tb, tw = fbw_costs(d_model=768, seq_len=1024)
    assert tw < tf < tb
    assert tb + tw == 2 * tf  # exact in the Table-1 flop model


# ---------------- solve_microbatches (zero-bubble) ----------------


def _ofob_bubble(p, m):
    """Classic 1F1B fill/drain bubble (p-1)/(m+p-1), as a reference only."""
    return (p - 1) / (m + p - 1)


def test_target_is_met_by_the_solved_schedule():
    # covers: SCHEDULER-I8
    sol = solve_microbatches(n_stages=6, bubble_target=0.05, fbw=(1.0, 1.0, 1.0))
    assert sol["family"] == "zb" and sol["bubble"] <= 0.05
    # and it is the smallest such M
    smaller = solve_microbatches(
        n_stages=6, bubble_target=0.05, fbw=(1.0, 1.0, 1.0), floor=sol["microbatches"] - 1
    )
    assert smaller["microbatches"] == sol["microbatches"]


def test_zero_bubble_needs_far_fewer_microbatches_than_1f1b():
    # WOULD-FAIL-IF: the solver priced the 1F1B fill/drain instead of the
    # schedule the runtime executes
    # covers: SCHEDULER-I8
    m = solve_microbatches(n_stages=6, bubble_target=0.05, fbw=(1.0, 1.0, 1.0))["microbatches"]
    m_1f1b = next(k for k in range(1, 10_000) if _ofob_bubble(6, k) <= 0.05)
    assert m * 4 < m_1f1b


def test_slow_links_need_more_microbatches():
    fast = solve_microbatches(n_stages=4, bubble_target=0.05, fbw=(1, 1, 1))["microbatches"]
    slow = solve_microbatches(n_stages=4, bubble_target=0.05, fbw=(1, 1, 1), comm_ratio=1.0)[
        "microbatches"
    ]
    assert slow > fast


def test_real_ratios_reach_the_target():
    sol = solve_microbatches(n_stages=6, bubble_target=0.05, fbw=fbw_costs(4096, 1024))
    assert sol["bubble"] <= 0.05


def test_token_cap_still_wins():
    # covers: SCHEDULER-I8
    sol = solve_microbatches(
        n_stages=8,
        bubble_target=0.001,
        tokens_per_step_cap=4 * 2048,
        mb_size=1,
        seq_len=2048,
        fbw=(1.0, 1.4, 0.6),
        comm_ratio=2.0,
    )
    assert sol["microbatches"] == 4
    assert "token step budget" in sol["reason"]


def test_single_stage_unchanged():
    # covers: SCHEDULER-I8
    sol = solve_microbatches(n_stages=1, fbw=(1, 1, 1))
    assert sol["microbatches"] == 1 and sol["bubble"] == 0.0


def test_floor_respected():
    # covers: SCHEDULER-I8
    sol = solve_microbatches(n_stages=6, bubble_target=0.5, fbw=(1, 1, 1), floor=40)
    assert sol["microbatches"] >= 40


def test_rank0_holds_the_deepest_stack():
    depths = [stage_live_graphs(6, r, k_inflight=1) for r in range(6)]
    assert depths == [6, 5, 4, 3, 2, 1]


def test_latency_cover_stretches_every_lifespan():
    base = [stage_live_graphs(6, r, 1) for r in range(6)]
    wan = [stage_live_graphs(6, r, 4) for r in range(6)]
    assert all(w == b + 3 for b, w in zip(base, wan))


# ---------------- rank ordering in the master ----------------


def test_rank_memory_weights_shape():
    from wantrain.model.gpt import GPTConfig
    from wantrain.runtime.spec import JobSpec
    from wantrain.scheduler.placement import rank_memory_weights as _rank_memory_weights

    spec = JobSpec(
        model=GPTConfig(n_layer=12, d_model=256, seq_len=256, vocab_size=50257),
        stages=4,
        mb_size=2,
    )
    w = _rank_memory_weights(spec, k_inflight=2)
    assert len(w) == 4
    # interior ranks are ordered by lifespan depth alone
    assert w[1] > w[2]
    # rank 0 outweighs every interior rank: deepest stack plus the embedding
    assert w[0] == max(w[0], w[1], w[2])
    # the last rank's fp32 logits (50k vocab) outweigh an interior rank's stack
    assert w[3] > w[2]


def test_master_matches_biggest_card_to_biggest_need():
    """Replicates run_master's zip of need-sorted ranks with vram-sorted workers."""
    from wantrain.model.gpt import GPTConfig
    from wantrain.runtime.spec import JobSpec
    from wantrain.scheduler.placement import rank_memory_weights as _rank_memory_weights

    spec = JobSpec(
        model=GPTConfig(n_layer=12, d_model=256, seq_len=256, vocab_size=50257),
        stages=3,
        mb_size=2,
    )
    weights = _rank_memory_weights(spec, k_inflight=2)
    workers = [
        {"name": "small", "vram_bytes": 8 * 2**30},
        {"name": "big", "vram_bytes": 24 * 2**30},
        {"name": "mid", "vram_bytes": 12 * 2**30},
    ]
    need = sorted(range(spec.stages), key=lambda r: -weights[r])
    have = sorted(range(len(workers)), key=lambda i: -workers[i]["vram_bytes"])
    slot_of = dict(zip(need, have))
    ordered = [workers[slot_of[r]] for r in range(spec.stages)]
    # with an untied 50k vocab the last rank is the heaviest slot; follow the weights
    assert need[0] == spec.stages - 1
    assert ordered[need[0]]["name"] == "big"
    assert ordered[need[-1]]["name"] == "small"


# ---------------- fleet ceiling ----------------


def test_fleet_ceiling_scales_with_batch_not_topology():
    from wantrain.scheduler.schedule import solve_fleet_ceiling

    a = solve_fleet_ceiling(40e6, 8192, util_target=0.5, fbw=(1, 1, 1), w_fill_depth=10, n_stages=8)
    b = solve_fleet_ceiling(80e6, 8192, util_target=0.5, fbw=(1, 1, 1), w_fill_depth=10, n_stages=8)
    assert b["fleet_max"] == pytest.approx(2 * a["fleet_max"], abs=2)
    c = solve_fleet_ceiling(40e6, 2048, util_target=0.5, fbw=(1, 1, 1), w_fill_depth=10, n_stages=8)
    assert c["fleet_max"] == pytest.approx(4 * a["fleet_max"], abs=4)


def test_fleet_ceiling_zb_dividend():
    from wantrain.scheduler.schedule import solve_fleet_ceiling

    full = solve_fleet_ceiling(40e6, 8192, fbw=(1, 1, 1), w_fill_depth=7, n_stages=8)
    none = solve_fleet_ceiling(40e6, 8192, fbw=(1, 1, 1), w_fill_depth=0, n_stages=8)
    assert full["zb_dividend"] == 3.0  # equal F/B/W, full battery
    assert none["zb_dividend"] == 1.0  # no fill, 1F1B economics
    assert full["fleet_max"] == pytest.approx(3 * none["fleet_max"], abs=3)


def test_fleet_ceiling_intra_split_and_latency_side():
    from wantrain.scheduler.schedule import solve_fleet_ceiling

    one = solve_fleet_ceiling(40e6, 2048, fbw=(1, 1, 1), w_fill_depth=7, n_stages=8)
    eight = solve_fleet_ceiling(
        40e6, 2048, fbw=(1, 1, 1), w_fill_depth=7, n_stages=8, intra_split=8
    )
    assert eight["fleet_max"] == pytest.approx(8 * one["fleet_max"], abs=8)
    assert one["fleet_latency_side"] is None and one["binding"] == "batch"
    # a slow, latency-bound fleet: 1T params, 140 hops of 100 ms, 66 TFLOP/s
    lat = solve_fleet_ceiling(
        40e6,
        2048,
        fbw=(1, 1, 1),
        w_fill_depth=139,
        n_stages=140,
        intra_split=8,
        n_params=1e12,
        rtt_s=0.1,
        flops_per_gpu=6.6e13,
    )
    assert lat["fleet_latency_side"] == pytest.approx(
        6 * 1e12 * 40e6 / (6.6e13 * 140 * 0.1 * 0.5), rel=1e-6
    )
    assert lat["fleet_max"] == min(lat["fleet_batch_side"], lat["fleet_latency_side"])
    tiny = solve_fleet_ceiling(
        40e6,
        2048,
        fbw=(1, 1, 1),
        w_fill_depth=7,
        n_stages=8,
        intra_split=8,
        n_params=1e9,
        rtt_s=0.2,
        flops_per_gpu=6.6e13,
    )
    assert tiny["binding"] == "latency" and "latency side binds" in tiny["reason"]


# ---------------- k stays honest ----------------


def test_k_inflight_unchanged_by_this_change():
    sol = solve_k_inflight(window_s=0.5, rtt_ms=200, boundary_bytes=2**20, link_mbps=50)
    assert sol["k"] >= 2


# ---------------- plan-wire (M4): GPTConfig hf and MoE plans end to end ----------------
# Census fixtures as the builder child writes them (HANDOUT-M4 I-10), with no
# transformers: a tensor shared by two units is listed once, at its first unit,
# and both ends of a tied table carry `tied`.

_HF_SEQ = 16
# a fast link: these tiny models compute for microseconds per microbatch
_MBPS, _RTT = 1e6, 5.0


def _census(V, h, n, heads, kv, hd, f, tied, softcap=None, final=None, windows=None, norms=2):
    shapes = [
        ["self_attn.q_proj.weight", [heads * hd, h]],
        ["self_attn.k_proj.weight", [kv * hd, h]],
        ["self_attn.v_proj.weight", [kv * hd, h]],
        ["self_attn.o_proj.weight", [h, heads * hd]],
        ["mlp.gate_proj.weight", [f, h]],
        ["mlp.up_proj.weight", [f, h]],
        ["mlp.down_proj.weight", [h, f]],
    ] + [[f"norm{i}.weight", [h]] for i in range(norms)]
    two_d = sum(a * b for _n, d in shapes if len(d) == 2 for a, b in [d])
    params = two_d + norms * h
    head = [["model.norm.weight", [h]]] + ([] if tied else [["lm_head.weight", [V, h]]])
    out = [
        dict(
            unit=0,
            kind="prologue",
            params=V * h,
            linear_macs_per_token=0,
            attn_heads=0,
            attn_kv_heads=0,
            head_dim=0,
            softcap=None,
            window=None,
            tied=tied,
            param_shapes=[["model.embed_tokens.weight", [V, h]]],
        )
    ]
    for i in range(n):
        out.append(
            dict(
                unit=i + 1,
                kind="layer",
                params=params,
                linear_macs_per_token=two_d,
                attn_heads=heads,
                attn_kv_heads=kv,
                head_dim=hd,
                softcap=softcap,
                window=(windows or [None] * n)[i],
                tied=False,
                param_shapes=[[f"model.layers.{i}.{k}", d] for k, d in shapes],
            )
        )
    out.append(
        dict(
            unit=n + 1,
            kind="epilogue",
            params=h + (0 if tied else V * h),
            linear_macs_per_token=V * h,
            attn_heads=0,
            attn_kv_heads=0,
            head_dim=0,
            softcap=final,
            window=None,
            tied=tied,
            param_shapes=head,
        )
    )
    return out


def _llama():
    """A tiny untied llama: V 96, h 32, 2 layers, 4 heads (2 kv, d 8), f 80."""
    hf_config = {
        "model_type": "llama",
        "vocab_size": 96,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 80,
        "tie_word_embeddings": False,
    }
    census = _census(96, 32, 2, 4, 2, 8, 80, False)
    fields = dict(n_layer=2, d_model=32, n_head=4, vocab_size=96, tie_embeddings=False)
    return hf_config, census, fields


def _gemma2():
    """A tiny gemma2: tied table (V 112, h 32), 4 layers, head_dim 16 (heads x
    hd = 64 != h), attention softcap 50, final softcap 30, sliding window 8 on
    every other layer, four norms per layer."""
    hf_config = {
        "model_type": "gemma2",
        "vocab_size": 112,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 64,
        "tie_word_embeddings": True,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "sliding_window": 8,
    }
    census = _census(
        112,
        32,
        4,
        4,
        2,
        16,
        64,
        True,
        softcap=50.0,
        final=30.0,
        windows=[8, None, 8, None],
        norms=4,
    )
    fields = dict(n_layer=4, d_model=32, n_head=4, vocab_size=112, tie_embeddings=True)
    return hf_config, census, fields


def _pins(census, fields):
    return {
        "hf_buffers": {"v": 1, "tensors": {}},
        "hf_census": census,
        "hf_code": {"transformers": "5.19.0", "torch": "fixture"},
        "fields": fields,
    }


def _hf_job(hf_config, **kw):
    return dict(arch="hf", hf_config=hf_config, seq_len=_HF_SEQ, **kw)


def _plan_fleet(n=4, tflops=80.0, vram=24 << 30):
    from wantrain.scheduler.planner import Fleet, Node

    nodes = [
        Node(f"n{i}", tflops=tflops, vram_bytes=vram, compute_cap=(8, 9), device_class=ONE_CLASS)
        for i in range(n)
    ]
    return Fleet(nodes=nodes, link_mbps=_MBPS, rtt_ms=_RTT, duplex="per_direction")


def _fleet_dict(job, n=2):
    return {
        "run_id": "r",
        "nodes": [
            dict(
                rank=i,
                host=f"h{i}",
                tflops=80.0,
                vram_bytes=24 << 30,
                compute_cap=[8, 9],
                device_class=ONE_CLASS,
            )
            for i in range(n)
        ],
        "link": {"mbps": _MBPS, "rtt_ms": _RTT, "duplex": "per_direction"},
        "job": dict({"steps": 3, "device": "cuda"}, **job),
    }


def _moe_job(**kw):
    """A native MoE job on the builtin block's dense fields (V 64, h 32, 4
    heads, S 16): 4 experts, k 2, f 48, a shared expert of 40 with its gate."""
    return dict(
        dict(
            vocab_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            seq_len=16,
            moe_experts=4,
            moe_top_k=2,
            moe_d_ff=48,
            moe_shared_d_ff=40,
        ),
        **kw,
    )


def _moe_cfg(**kw):
    from wantrain.scheduler import fleetplan

    cfg, _notes = fleetplan.model_config(_moe_job(**kw))
    return cfg


def test_m4_plan_wire_model_keys_and_refused_pins():
    """I-14: MODEL_KEYS carries hf_config, hf_impl and every moe_* field of
    GPTConfig (SPEC-model §2); a plan dict carrying hf_buffers, hf_census or
    hf_code is refused naming the key, whatever its arch; an hf plan without
    hf_pins is refused naming the census."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: MODEL_KEYS lacks a moe_* or hf key, a caller-written pin
    # reaches the model, or an hf plan is planned without the builder's census.
    import dataclasses

    from wantrain.model.gpt import GPTConfig
    from wantrain.scheduler import fleetplan

    moe = {f.name for f in dataclasses.fields(GPTConfig) if f.name.startswith("moe_")}
    assert len(moe) == 23 and moe <= set(fleetplan.MODEL_KEYS)
    assert {"hf_config", "hf_impl"} <= set(fleetplan.MODEL_KEYS)
    assert not {"hf_buffers", "hf_census", "hf_code"} & set(fleetplan.MODEL_KEYS)
    hf_config, census, fields = _llama()
    for key in ("hf_buffers", "hf_census", "hf_code"):
        for job in (_hf_job(hf_config), {"model": "toy"}):
            with pytest.raises(fleetplan.PlanRefused, match=key):
                fleetplan.model_config(dict(job, **{key: census}), _pins(census, fields))
    with pytest.raises(fleetplan.PlanRefused, match="hf_census"):
        fleetplan.model_config(_hf_job(hf_config))
    with pytest.raises(fleetplan.PlanRefused, match="hf_census"):
        fleetplan.build_spec(_fleet_dict(_hf_job(hf_config)))
    # the control: the same plan with the builder's pins is planned
    assert fleetplan.build_spec(_fleet_dict(_hf_job(hf_config)), hf_pins=_pins(census, fields))


def test_m4_plan_wire_hf_pins_set_and_mirror_refused():
    """model_config(job, hf_pins) sets hf_buffers, hf_census and hf_code from
    the pins and the mirrored fields from the census fields (not the
    preset's); a plan dict whose mirror differs is refused naming the field,
    one that agrees is accepted; pins missing a key are refused naming it."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the pins or the mirrored fields are not set, or a plan-dict
    # mirror that differs from the census is silently overwritten.
    from wantrain.scheduler import fleetplan

    hf_config, census, fields = _gemma2()
    pins = _pins(census, fields)
    cfg, _notes = fleetplan.model_config(_hf_job(hf_config), pins)
    assert cfg.hf_census is census and cfg.hf_buffers is pins["hf_buffers"]
    assert cfg.hf_code is pins["hf_code"] and cfg.hf_config == hf_config
    # the toy preset is V 256, 6 layers, h 128: every mirrored field moved
    assert (cfg.vocab_size, cfg.n_layer, cfg.d_model, cfg.n_head, cfg.tie_embeddings) == (
        112,
        4,
        32,
        4,
        True,
    )
    cfg.validate()
    for key, bad in (
        ("n_layer", 5),
        ("d_model", 64),
        ("vocab_size", 128),
        ("tie_embeddings", False),
    ):
        with pytest.raises(fleetplan.PlanRefused, match=key):
            fleetplan.model_config(_hf_job(hf_config, **{key: bad}), pins)
    agree, _n = fleetplan.model_config(_hf_job(hf_config, n_layer=4, vocab_size=112), pins)
    assert agree.n_layer == 4
    for key in ("hf_buffers", "hf_census", "hf_code", "fields"):
        broken = {k: v for k, v in pins.items() if k != key}
        with pytest.raises(fleetplan.PlanRefused, match=key):
            fleetplan.model_config(_hf_job(hf_config), broken)


def _write_safetensors(path, tensors):
    """A minimal safetensors file (8-byte header length, JSON header, data)."""
    import json
    import struct

    header, data = {}, b""
    for name, (shape, values) in tensors.items():
        raw = struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + data)


def test_m4_plan_wire_hf_checkpoint_config_adopted(tmp_path):
    """A fine-tune's checkpoint decides: its config.json becomes cfg.hf_config
    and its digest is taken here through hf.checkpoint_digest (the native
    config_from_hf is gone); a plan hf_config that differs is refused."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the planner still calls config_from_hf, keeps the plan's
    # hf_config over the checkpoint's, or trusts a caller-given digest.
    import json

    from wantrain.model import hf
    from wantrain.scheduler import fleetplan

    hf_config, census, fields = _llama()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(hf_config))
    _write_safetensors(ckpt / "model.safetensors", {"w": ([2, 2], [1.0, 2.0, 3.0, 4.0])})
    job = dict(arch="hf", hf_model=str(ckpt), hf_model_sha256="0" * 64, seq_len=_HF_SEQ)
    cfg, notes = fleetplan.model_config(job, _pins(census, fields))
    assert not hasattr(hf, "config_from_hf")
    assert cfg.hf_config == hf_config
    assert cfg.hf_model_sha256 == hf.checkpoint_digest(str(ckpt)) != "0" * 64
    assert any("config.json" in n for n in notes)
    same, _n = fleetplan.model_config(dict(job, hf_config=dict(hf_config)), _pins(census, fields))
    assert same.hf_config == hf_config
    with pytest.raises(fleetplan.PlanRefused, match="hf_config"):
        fleetplan.model_config(
            dict(job, hf_config=dict(hf_config, vocab_size=97)), _pins(census, fields)
        )


def test_m4_plan_wire_corpus_vocab_against_census_fields(tmp_path):
    """The corpus-vocab check of an hf plan compares the corpus against the
    census fields' vocabulary: equal is planned, different is refused; a
    dense plan still adopts the corpus vocabulary (unchanged)."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: an hf plan adopts the corpus vocabulary over the
    # checkpoint's embedding, or compares against the preset's.
    from wantrain.model import dataset as ds
    from wantrain.scheduler import fleetplan

    hf_config, census, fields = _llama()
    good, bad = str(tmp_path / "good.bin"), str(tmp_path / "bad.bin")
    ds.write_manifest(good, "tok", 96, 1000, 1)
    ds.write_manifest(bad, "tok", 256, 1000, 1)  # the toy preset's vocabulary
    cfg, _n = fleetplan.model_config(_hf_job(hf_config, dataset=good), _pins(census, fields))
    assert cfg.vocab_size == 96
    with pytest.raises(fleetplan.PlanRefused, match="vocab 256.*census fields have vocab 96"):
        fleetplan.model_config(_hf_job(hf_config, dataset=bad), _pins(census, fields))
    dense, notes = fleetplan.model_config({"model": "toy", "dataset": good})
    assert dense.vocab_size == 96 and any("vocab_size 256 -> 96" in n for n in notes)


def test_m4_plan_wire_hf_plan_charges_i22():
    """plan_job on a GPTConfig arch 'hf' built from census fixtures charges
    SCHEDULER-I22's terms with scheduler-m4's pure functions' numbers: the
    fp32 cut, each stage's census parameters (a split tied table on the
    Head), and for the tied gemma2 the width-1 step carries ΔT_tied equal to
    autoconfig.delta_t_tied on the same inputs; without a census it is
    refused naming it."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: plan_job prices an hf GPTConfig with the dense block's
    # costs, a bf16 cut, the tied table on stage 0, or leaves ΔT_tied out.
    import math
    import types

    from wantrain.scheduler import autoconfig, fleetplan
    from wantrain.scheduler.planner import plan_job
    from wantrain.scheduler.zero_bubble import analytic_costs

    fleet = _plan_fleet()
    hf_config, census, fields = _llama()
    cfg, _n = fleetplan.model_config(
        _hf_job(hf_config, compute_dtype="bf16"), _pins(census, fields)
    )
    out = plan_job(cfg, fleet, steps=10, device="cuda")
    spec = out["spec"]
    assert spec is not None and out["rationale"]["boundary_dtype"] == "fp32"
    assert spec.model.boundary_dtype == "fp32" and spec.model.compute_dtype == "bf16"
    want = [sum(census[u]["params"] for u in range(a, b)) for a, b in spec.partition]
    assert out["rationale"]["stage_params"] == want
    assert spec.stage_widths == [1] * spec.stages

    hf_config, census, fields = _gemma2()
    cfg, _n = fleetplan.model_config(_hf_job(hf_config), _pins(census, fields))
    out = plan_job(cfg, fleet, steps=10, device="cuda")
    spec = out["spec"]
    assert spec is not None and spec.stages == 4 and spec.tied_relay()
    V, h = 112, 32
    part = spec.partition
    want = [sum(census[u]["params"] for u in range(a, b)) for a, b in part]
    want[0] -= V * h  # the split table trains on the Head (SCHEDULER-I22 §2)
    want[-1] += V * h
    assert out["rationale"]["stage_params"] == want
    w1 = next(c for c in out["candidates"] if c["w"] == 1)
    tied = w1["detail"]["tied"]
    tokens = spec.mb_size * _HF_SEQ
    ns = types.SimpleNamespace(model=cfg, recompute=spec.recompute, partition=[part[0]])
    t_f, t_b, t_w = (x[0] * tokens / 80e12 for x in analytic_costs(ns))
    rows = min(V, tokens)
    expect = autoconfig.delta_t_tied(
        t_f + t_b + t_w,
        t_f,
        t_b,
        [rows] * spec.microbatches,
        [rows] * spec.microbatches,
        len(part) - 1,
        _RTT / 2000.0,
        _MBPS * 1e6 / 8,
        d_model=h,
        vocab_size=V,
        tokens_mb=tokens,
        mem_bw_Bps=math.inf,
    )
    assert tied["delta_t_tied"] == expect and expect["total"] > 0
    assert tied["fwd_bytes_per_mb"] == autoconfig.tied_hop_bytes(rows, rows, h)[0]
    assert w1["step_s"] >= expect["total"]
    assert out["rationale"]["tied"]["hops"] == [[0, 1], [1, 2], [2, 3]]

    cfg.hf_census = None
    with pytest.raises(ValueError, match="hf_census"):
        plan_job(cfg, fleet, steps=10, device="cuda")


def _real_block_params(cfg, layer, active):
    """A built MoE block's GEMM parameters: the attention's weights plus the
    router and the experts (k of them when `active`, all E otherwise), the
    shared expert and its gate, counted from the real module."""
    attn = sum(
        p.numel()
        for n, p in layer.named_parameters()
        if n.endswith(".weight") and n.split(".")[0] in ("qkv", "q", "kv", "proj")
    )
    mlp = layer.mlp
    one = sum(p.numel() for p in mlp.experts[0].parameters())
    routed = (cfg.moe_top_k if active else cfg.moe_experts) * one
    rest = sum(p.numel() for n, p in mlp.named_parameters() if not n.startswith("experts."))
    return attn + routed + rest


def test_m4_plan_wire_moe_plan_charges_i21(monkeypatch):
    """plan_job on a GPTConfig with moe_experts > 0 charges SCHEDULER-I21's
    terms: each stage's parameters count all E experts of the real modules
    (bytes and memory at total), the cut balances active compute; a
    plan-mode MoE job through build_spec gives a spec whose model holds
    every moe_* key of the dict, and a MODEL_KEYS without them trains dense.
    Dense plans are unchanged by the new keys."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: MODEL_KEYS drops a moe_* key and the run trains dense, or
    # an MoE stage's memory is charged at the active parameters.
    from wantrain.model.gpt import build_layers
    from wantrain.scheduler import fleetplan
    from wantrain.scheduler.planner import plan_job

    cfg = _moe_cfg()
    layers = build_layers(cfg, 0)
    out = plan_job(cfg, _plan_fleet(2), steps=10, device="cuda")
    spec = out["spec"]
    assert spec is not None and spec.model.moe_experts == 4
    total = _real_block_params(cfg, layers[1], active=False)
    active = _real_block_params(cfg, layers[1], active=True)
    assert total > active
    for (a, b), got in zip(spec.partition, out["rationale"]["stage_params"]):
        blocks = sum(1 for j in range(max(a, 1), min(b, 3)))
        emb = 64 * 32 * ((a == 0) + (b == 4))
        assert got == blocks * total + emb
    job = _moe_job(
        moe_router_bias=True,
        moe_lbl_coef=0.01,
        moe_z_coef=0.001,
        moe_bias_rate=0.002,
        moe_layers=[1],
        moe_score="sigmoid",
        moe_select="score_bias",
        moe_norm=True,
    )
    built = fleetplan.build_spec(_fleet_dict(job))
    model = built["spec"].model
    moe_keys = [k for k in job if k.startswith("moe_")]
    assert len(moe_keys) == 12
    assert {k: getattr(model, k) for k in moe_keys} == {k: job[k] for k in moe_keys}
    # the deliberately wrong implementation: the pre-M4 MODEL_KEYS trains dense
    pre_m4 = tuple(
        k
        for k in fleetplan.MODEL_KEYS
        if not k.startswith("moe_") and k not in ("hf_config", "hf_impl")
    )
    with monkeypatch.context() as mp:
        mp.setattr(fleetplan, "MODEL_KEYS", pre_m4)
        dense_run = fleetplan.build_spec(_fleet_dict(job))["spec"].model
        dense_job = fleetplan.build_spec(_fleet_dict({"model": "toy"}))["spec_json"]
    assert dense_run.moe_experts == 0
    assert {k: getattr(dense_run, k) for k in moe_keys} != {k: job[k] for k in moe_keys}
    # dense plans: the new keys change nothing
    assert fleetplan.build_spec(_fleet_dict({"model": "toy"}))["spec_json"] == dense_job


def test_m4_plan_wire_moe_rationale_records_c_s_and_profiled_time():
    """SCHEDULER-I21's rationale clause: a plan of a GPTConfig with
    moe_experts > 0 records, for every stage holding an MoE block, its
    analytic c_s = 8·P^F_s·T_mb/F next to its profiled per-microbatch time
    (keys c_s_analytic and t_profiled, seconds), P^F_s at the active
    parameters of the real modules; and names ρ_s for every stage of every
    candidate pieces are refused for."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: the rationale drops a stage's c_s or its profiled time,
    # or charges c_s at all E experts.
    import types

    from wantrain.model.gpt import build_layers
    from wantrain.scheduler.planner import plan_job
    from wantrain.scheduler.zero_bubble import analytic_costs

    cfg = _moe_cfg(n_layer=4)
    layers = build_layers(cfg, 0)
    active = _real_block_params(cfg, layers[1], active=True)
    total = _real_block_params(cfg, layers[1], active=False)
    for n, tflops in ((2, [80.0, 40.0]), (4, [80.0] * 4)):
        fleet = _plan_fleet(n)
        for node, t in zip(fleet.nodes, tflops):
            node.tflops = t
        out = plan_job(cfg, fleet, steps=10, device="cuda", allow_pieces=True)
        spec = out["spec"]
        assert spec is not None
        rows = out["rationale"]["moe_compute"]
        moe_stages = [s for s, (a, b) in enumerate(spec.partition) if max(a, 1) < min(b, 5)]
        assert [r["stage"] for r in rows] == moe_stages and moe_stages
        tokens = spec.mb_size * cfg.seq_len
        for r in rows:
            a, b = spec.partition[r["stage"]]
            peers = out["placement"][r["stage"]]
            rate = sum(fleet.nodes[p["node"]].tflops for p in peers) / len(peers) * 1e12
            blocks = max(0, min(b, 5) - max(a, 1))
            head = 64 * 32 if b == 6 else 0
            assert r["c_s_analytic"] == pytest.approx(8 * (blocks * active + head) * tokens / rate)
            wrong = 8 * (blocks * total + head) * tokens / rate
            assert r["c_s_analytic"] != pytest.approx(wrong)
            ns = types.SimpleNamespace(model=cfg, recompute=spec.recompute, partition=[(a, b)])
            want_t = sum(x[0] for x in analytic_costs(ns)) * tokens / rate
            assert r["t_profiled"] == pytest.approx(want_t) and r["t_profiled"] > 0
    rho = out["rationale"]["moe_rho"]
    refused = [
        c for c in out["candidates"] if c["shard"] == "pieces" and c["w"] > 1 and not c["admitted"]
    ]
    assert set(rho) == {c["name"] for c in refused}
    for c in refused:
        assert len(rho[c["name"]]) == c["S"] and all(x > 1 for x in rho[c["name"]])


def _runtime_units(cfg, li):
    from wantrain.model.gpt import build_layers
    from wantrain.runtime import pieces

    layers = build_layers(cfg, 0)
    units = pieces.exchange_units([layers[li]])
    params = [n for n, _p in layers[li].named_parameters()]
    return units, params


def test_m4_plan_wire_moe_units_equal_the_runtime():
    """BA-466 on real configs: the planner's units of one native MoE block
    equal pieces.exchange_units of the same block built by gpt.build_layers,
    names and byte sizes, in order (the small tensors with the selection
    biases first, the attention, one unit per expert projection, the shared
    expert and its gate), and each tensor's ZeRO-1 index is its position in
    the block's parameter list (the biases after it); for the default and a
    clamped, router-biased, GQA block. An hf GPTConfig plan is width 1 with
    every wide candidate refused 'unpriced' (D-10)."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: the planner's MoE units drift from the runtime's.
    from wantrain.scheduler import autoconfig, fleetplan
    from wantrain.scheduler.planner import _stage_units, moe_block_units, plan_job

    variants = (
        {},
        dict(
            moe_expert="swiglu_clamped",
            moe_router_bias=True,
            n_kv_head=2,
            moe_top_k=1,
            moe_shared_d_ff=0,
        ),
    )
    for kw in variants:
        cfg = _moe_cfg(**kw)
        runtime, params = _runtime_units(cfg, 1)
        names, _order = moe_block_units(cfg, autoconfig.moe_fields(cfg))
        got = [[(n, e) for n, e in u] for u in names]
        want = [[(n.split(".", 1)[1], p.numel()) for n, p in u] for u in runtime]
        assert got == want
        units, idx, blocks = _stage_units(cfg, 1, 2, 4, 4)
        assert blocks == 1
        assert [u.params * u.b_p for u in units] == [
            sum(p.numel() * p.element_size() for _n, p in u) for u in runtime
        ]
        E = cfg.moe_experts
        assert [u.lane for u in units] == [E * 8] + [0] * (len(units) - 1)
        for u, (tensors, rt) in enumerate(zip(idx, runtime)):
            pos = []
            for n, _p in rt:
                local = n.split(".", 1)[1]
                pos.append(len(params) if local == "mlp.sel_bias" else params.index(local))
            assert [i for _e, i in tensors] == pos, u
            assert [e for e, _i in tensors] == [p.numel() for _n, p in rt]
    # the dense block keeps the M2b unit model (dense plans unchanged)
    dense, _i, _b = _stage_units(_moe_cfg(moe_experts=0), 1, 2, 2, 2)
    assert [u.params for u in dense] == [3072, 1024, 4096, 4096]

    hf_config, census, fields = _llama()
    cfg, _n = fleetplan.model_config(_hf_job(hf_config), _pins(census, fields))
    out = plan_job(cfg, _plan_fleet(), steps=10, device="cuda", allow_pieces=True)
    assert out["spec"] is not None and out["stage_widths"] == [1] * out["spec"].stages
    wide = [c for c in out["candidates"] if c["w"] > 1]
    assert wide
    for c in wide:
        assert not c["admitted"] and any(r["kind"] == "unpriced" for r in c["reasons"])


@pytest.mark.parametrize("piece", ["scheduler-6", "runtime-7a", "runtime-7b"])
def test_m4_plan_wire_rollback_refuses_every_split_of_a_tied_hf_table(monkeypatch, piece):
    """BA-467: with a tied piece rolled back (model/rollback.py, read at call
    time), plan_job on the tied gemma2 marks every candidate that splits the
    table not admitted, naming the piece and 'tied-split-refused', and plans
    the job on one stage when one fits (refused MEM when none does); with
    ROLLED_BACK empty the split candidates are priced with I22's tied terms."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a rolled-back relay is still planned, or the one-stage
    # plan is refused MEM on the fastest node while a slower node fits.
    from wantrain.model import rollback
    from wantrain.scheduler import fleetplan
    from wantrain.scheduler.planner import plan_job

    hf_config, census, fields = _gemma2()

    def cfg():
        return fleetplan.model_config(_hf_job(hf_config), _pins(census, fields))[0]

    live = plan_job(cfg(), _plan_fleet(), steps=10, device="cuda")
    assert live["spec"].stages == 4 and live["spec"].tied_relay()
    w1 = next(c for c in live["candidates"] if c["w"] == 1)
    assert w1["admitted"] and w1["detail"]["tied"]["delta_t_tied"]["total"] > 0

    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({piece}))
    for n in (4, 2):
        out = plan_job(cfg(), _plan_fleet(n), steps=10, device="cuda")
        split = [c for c in out["candidates"] if c["S"] > 1]
        assert split
        for c in split:
            assert not c["admitted"]
            texts = [r["text"] for r in c["reasons"] if r["kind"] == "tied"]
            assert texts and all(piece in t and "tied-split-refused" in t for t in texts)
        one = next(c for c in out["candidates"] if c["S"] == 1)
        assert one["admitted"] and out["rationale"]["chosen"] == "S 1 w 1"
        spec = out["spec"]
        assert spec.stages == 1 and spec.partition == [[0, 6]] and not spec.tied_relay()
        assert out["stage_widths"] == [1] and len(out["placement"]) == 1
    # Gemma2-2B's shape (V 256000, h 2304, 26 layers: ~2.6e9 parameters, ~42 GB
    # of state at 16 B each): four 24 GB stages fit, one stage does not
    big = _census(
        256000,
        2304,
        26,
        8,
        4,
        256,
        9216,
        True,
        softcap=50.0,
        final=30.0,
        windows=[4096, None] * 13,
        norms=4,
    )
    big_fields = dict(n_layer=26, d_model=2304, n_head=8, vocab_size=256000, tie_embeddings=True)
    big_cfg, _n = fleetplan.model_config(
        _hf_job(
            dict(
                hf_config,
                vocab_size=256000,
                hidden_size=2304,
                num_hidden_layers=26,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=256,
                intermediate_size=9216,
                sliding_window=4096,
            )
        ),
        _pins(big, big_fields),
    )
    small = plan_job(big_cfg, _plan_fleet(4), steps=10, device="cuda")
    assert small["spec"] is None
    one = next(c for c in small["candidates"] if c["S"] == 1)
    assert not one["admitted"] and [r["kind"] for r in one["reasons"]] == ["MEM"]
    assert all(not c["admitted"] for c in small["candidates"])
    # a heterogeneous fleet: the fastest nodes cannot hold the whole model, a
    # slower 160 GB node can, so the job is planned on it (one fits)
    hetero = _plan_fleet(4)
    hetero.nodes[3].tflops, hetero.nodes[3].vram_bytes = 40.0, 160 << 30
    big_out = plan_job(big_cfg, hetero, steps=10, device="cuda")
    one = next(c for c in big_out["candidates"] if c["S"] == 1)
    assert one["admitted"] and one["nodes"] == [[3]]
    assert big_out["spec"].stages == 1 and big_out["placement"][0][0]["node"] == 3


def test_m4_plan_wire_rollback_one_stage_keeps_a_pinned_partition(monkeypatch):
    """BA-467 under an operator's pinned partition: the one-stage candidate
    moves the cut, so it is refused naming the pin like every wide one (the
    override wins, never dropped); the pinned split stays refused naming the
    rolled-back piece, and no plan leaves the planner."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: under rollback the one-stage plan silently replaces the
    # job's pinned partition.
    from wantrain.model import rollback
    from wantrain.scheduler import fleetplan
    from wantrain.scheduler.planner import Overrides, plan_job

    hf_config, census, fields = _gemma2()
    cfg = fleetplan.model_config(_hf_job(hf_config), _pins(census, fields))[0]
    # the operator pins the cut the live planner chooses (four stages, the table split)
    live = plan_job(cfg, _plan_fleet(4), steps=10, device="cuda")
    pin = Overrides(partition=[list(x) for x in live["spec"].partition])
    assert len(pin.partition) == 4 and live["spec"].tied_relay()
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"scheduler-6"}))
    out = plan_job(cfg, _plan_fleet(4), steps=10, device="cuda", overrides=pin)
    assert out["spec"] is None
    one = next(c for c in out["candidates"] if c["S"] == 1)
    assert not one["admitted"]
    assert [(r["kind"], r["text"]) for r in one["reasons"]] == [
        ("override", "override: partition pinned by the job")
    ]
    split = [c for c in out["candidates"] if c["S"] > 1]
    assert split and all(not c["admitted"] for c in split)


_NO_TRANSFORMERS = r"""
import importlib.abc, json, sys

seen = []


class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "transformers" or name.startswith("transformers.") or name == "wantrain.model.hfm":
            seen.append(name)
            raise ImportError("refused: " + name)
        return None


sys.meta_path.insert(0, Refuse())
data = json.load(open(sys.argv[1]))
from wantrain.scheduler import fleetplan
from wantrain.scheduler.planner import Fleet, Node, plan_job

cfg, _ = fleetplan.model_config(data["job"], data["pins"])
nodes = [Node("n%d" % i, tflops=80.0, vram_bytes=24 << 30, compute_cap=(8, 9),
              device_class="cuda:sim") for i in range(4)]
out = plan_job(cfg, Fleet(nodes=nodes, link_mbps=1e6, rtt_ms=5.0, duplex="per_direction"),
               steps=10, device="cuda", allow_pieces=True)
built = fleetplan.build_spec(data["fleet"], hf_pins=data["pins"])
loaded = sorted(m for m in sys.modules if m.split(".")[0] == "transformers" or m == "wantrain.model.hfm")
planned = list(seen)
try:
    import transformers  # noqa: F401  (the control: the finder is live)
except ImportError:
    pass
print(json.dumps({"planned": planned, "loaded": loaded, "control": seen,
                  "stages": [out["spec"].stages, built["spec"].stages]}))
"""


def test_m4_plan_wire_planning_never_imports_transformers(tmp_path):
    """BA-441: a subprocess with a meta-path finder that records and refuses
    any import of transformers or wantrain.model.hfm plans GPTConfig hf jobs
    (the tied gemma2 and the llama) from census fixtures through plan_job and
    build_spec(hf_pins=...): nothing is recorded or loaded; the finder's own
    control records an import, so the test holds whether or not transformers
    is installed."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the planner or finalize imports transformers (or hfm).
    import json
    import subprocess
    import sys

    for fixture in (_gemma2, _llama):
        hf_config, census, fields = fixture()
        job = _hf_job(hf_config)
        data = {"job": job, "pins": _pins(census, fields), "fleet": _fleet_dict(job, n=4)}
        path = tmp_path / "in.json"
        path.write_text(json.dumps(data))
        run = subprocess.run(
            [sys.executable, "-c", _NO_TRANSFORMERS, str(path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert run.returncode == 0, run.stderr[-4000:]
        got = json.loads(run.stdout.strip().splitlines()[-1])
        assert got["planned"] == [] and got["loaded"] == []
        assert got["control"] == ["transformers"]
        assert got["stages"][0] >= 2 and got["stages"][1] >= 2
