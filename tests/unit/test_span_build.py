"""A span build (RUNTIME-I22) keeps only its entries, built from the same draws as the full build: every kept tensor
equals the full build's bit for bit, the generator ends in the same state, and nothing outside the span stays.

WOULD-FAIL-IF: skipping a layer skipped its draws (every later layer would shift), gpt2_init's draws for a dropped
layer were not replayed, a tied table were split, or a dropped layer stayed resident.
"""

import pytest
import torch

from wantrain.model.gpt import GPTConfig, build_layers, make_config

MOE = dict(
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_experts=4,
    moe_top_k=2,
    moe_d_ff=48,
    moe_score="sigmoid",
    moe_select="score_bias",
    moe_norm=True,
    moe_bias_rate=0.001,
)
SMALL = dict(vocab_size=97, d_model=32, n_layer=4, n_head=4, seq_len=16)


def _configs():
    yield "builtin", make_config("toy")
    yield "moe", GPTConfig(**SMALL, **MOE)
    yield "general-gpt2", GPTConfig(**SMALL, norm="rmsnorm", mlp="swiglu", init_scheme="gpt2")


@pytest.mark.parametrize("name,cfg", list(_configs()), ids=[n for n, _ in _configs()])
def test_a_span_build_is_the_full_build_s_slice(name, cfg):
    # covers: RUNTIME-I22
    full = build_layers(cfg, 7)
    after_full = torch.get_rng_state()
    n = len(full)
    for a, b in [(0, 2), (1, n - 1), (n - 2, n), (0, n)]:
        part = build_layers(cfg, 7, span=(a, b))
        assert torch.equal(torch.get_rng_state(), after_full)
        assert len(part) == n
        for i in range(n):
            if a <= i < b:
                want = full[i].state_dict()
                got = part[i].state_dict()
                assert want.keys() == got.keys()
                assert all(torch.equal(want[k], got[k]) for k in want)
            else:
                assert isinstance(part[i], torch.nn.Identity)


def test_a_tied_table_builds_in_full():
    # covers: RUNTIME-I22
    cfg = GPTConfig(**SMALL, norm="rmsnorm", mlp="swiglu", init_scheme="gpt2", tie_embeddings=True)
    part = build_layers(cfg, 7, span=(1, 3))
    assert not any(isinstance(x, torch.nn.Identity) for x in part)
