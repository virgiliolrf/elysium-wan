"""optimizer "adamw-host" (RUNTIME-I20): AdamW with its moments in host memory,
written in single-rounding ops so its bits depend on neither the chunking nor
the thread count, checkpointed in torch's layout under its own recipe name.

WOULD-FAIL-IF: the update fused a multiply into an add or divided by a scalar
(chunk or thread count would show), the moments lived on the device, a resume
did not continue bit for bit, a torch-AdamW checkpoint loaded into it silently,
or the recipe pin and the reference's digest did not name the optimizer.
"""

import pytest
import torch

from wantrain.launch.local import recipe_digest
from wantrain.model.gpt import make_config
from wantrain.runtime import optim
from wantrain.runtime.spec import ADAMW, JobSpec


def _params(seed=3):
    g = torch.Generator().manual_seed(seed)
    shapes = [(37, 19), (19,), (5, 3, 7), (1001,)]
    return [torch.randn(s, generator=g).requires_grad_(True) for s in shapes]


def _grads(params, step, seed=5):
    g = torch.Generator().manual_seed(seed + step)
    return [torch.randn(p.shape, generator=g) * 10.0 ** (-step % 4) for p in params]


def _train(opt_factory, steps=4, params=None):
    params = params or _params()
    opt = opt_factory(params)
    for step in range(steps):
        for p, g in zip(params, _grads(params, step)):
            p.grad = g
        opt.step()
        opt.zero_grad(set_to_none=True)
    return params, opt


def _bits(ts):
    return [t.detach().clone() for t in ts]


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_chunking_does_not_move_a_bit():
    # covers: RUNTIME-I20
    big, _ = _train(lambda ps: optim.HostAdamW(ps, lr=1e-3, **ADAMW))
    small, _ = _train(lambda ps: optim.HostAdamW(ps, lr=1e-3, chunk=7, **ADAMW))
    assert _equal(_bits(big), _bits(small))


def test_thread_count_does_not_move_a_bit():
    # covers: RUNTIME-I20
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one, _ = _train(lambda ps: optim.HostAdamW(ps, lr=1e-3, chunk=64, **ADAMW))
        torch.set_num_threads(max(2, before))
        many, _ = _train(lambda ps: optim.HostAdamW(ps, lr=1e-3, chunk=64, **ADAMW))
    finally:
        torch.set_num_threads(before)
    assert _equal(_bits(one), _bits(many))


def test_it_is_adamw():
    # covers: RUNTIME-I20
    # Same update as torch's AdamW up to the last place; a wrong moment, bias
    # correction or decay would be far outside this tolerance.
    host, _ = _train(lambda ps: optim.HostAdamW(ps, lr=1e-2, **dict(ADAMW, weight_decay=0.1)))
    ref, _ = _train(lambda ps: torch.optim.AdamW(ps, lr=1e-2, **dict(ADAMW, weight_decay=0.1)))
    for a, b in zip(host, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_moments_live_in_host_memory_and_checkpoint_in_torch_layout():
    # covers: RUNTIME-I20
    params, opt = _train(lambda ps: optim.build("adamw-host", ps, 1e-3))
    sd = opt.state_dict()
    ref = torch.optim.AdamW(_params(), lr=1e-3, **ADAMW).state_dict()
    group = sd["param_groups"][0]
    assert group["params"] == list(range(len(params)))
    assert group.pop("moments") == "host"
    assert set(group) == set(ref["param_groups"][0])
    for i, p in enumerate(params):
        st = sd["state"][i]
        assert set(st) == {"step", "exp_avg", "exp_avg_sq"}
        assert float(st["step"]) == 4.0 and st["step"].dtype == torch.float32
        assert st["exp_avg"].device.type == "cpu" and st["exp_avg"].shape == p.shape
    assert optim.hparams(1e-3, "adamw-host")["moments"] == "host"
    assert "moments" not in optim.hparams(1e-3)


def test_resume_continues_bit_for_bit():
    # covers: RUNTIME-I20
    full, _ = _train(lambda ps: optim.build("adamw-host", ps, 1e-3), steps=6)
    part, opt = _train(lambda ps: optim.build("adamw-host", ps, 1e-3), steps=3)
    sd = opt.state_dict()
    resumed = [p.detach().clone().requires_grad_(True) for p in part]
    opt2 = optim.build("adamw-host", resumed, 1e-3)
    opt2.load_state_dict(sd)
    for step in range(3, 6):
        for p, g in zip(resumed, _grads(resumed, step)):
            p.grad = g
        opt2.step()
        opt2.zero_grad()
    assert _equal(_bits(full), _bits(resumed))


def test_state_of_another_recipe_is_refused():
    # covers: RUNTIME-I20
    _, torch_opt = _train(lambda ps: torch.optim.AdamW(ps, lr=1e-3, **ADAMW), steps=1)
    host = optim.build("adamw-host", _params(), 1e-3)
    with pytest.raises(ValueError, match="another recipe"):
        host.load_state_dict(torch_opt.state_dict())
    with pytest.raises(ValueError, match="fp32"):
        optim.HostAdamW([torch.zeros(3, dtype=torch.bfloat16, requires_grad=True)], lr=1e-3)


def test_spec_pins_and_reference_digest_name_the_optimizer():
    # covers: RUNTIME-I20
    base = JobSpec(model=make_config("toy"), stages=2)
    host = JobSpec(model=make_config("toy"), stages=2, optimizer="adamw-host")
    assert "optimizer" not in base.finalize().pins()
    assert host.finalize().pins()["optimizer"] == "adamw-host"
    assert recipe_digest(base) != recipe_digest(host)
    with pytest.raises(ValueError, match="not one of"):
        JobSpec(model=make_config("toy"), stages=2, optimizer="sgd").finalize()
    with pytest.raises(ValueError, match="pieces"):
        JobSpec(
            model=make_config("toy"), stages=2, optimizer="adamw-host", shard="pieces"
        ).finalize()
    assert optim.state_bytes_per_param("adamw-host") == 8
    assert optim.state_bytes_per_param() == 16


def test_the_planner_prices_host_moments_at_eight_bytes():
    # covers: RUNTIME-I20
    # 780M trainable elements on 14 GiB: torch's AdamW (16 B each) does not fit, host moments (8 B) do.
    from wantrain.scheduler.planner import stage_fits

    cfg = make_config("gpt760m")
    vram = 14 * 2**30
    assert not stage_fits(cfg, 0, cfg.n_layer + 2, vram)
    assert stage_fits(
        cfg, 0, cfg.n_layer + 2, vram, state_bpp=optim.state_bytes_per_param("adamw-host")
    )


def test_staged_moments_do_not_move_a_bit():
    # covers: RUNTIME-I24
    # WOULD-FAIL-IF: the staging slots (chunk k in slot k % slots, refilled across parameter boundaries) read or
    # wrote a moment chunk out of place: the staged run's parameters or moments differ from the direct run's.
    direct, d_opt = _train(
        lambda ps: optim.HostAdamW(ps, lr=1e-3, chunk=7, pipeline=False, **ADAMW)
    )
    staged, s_opt = _train(
        lambda ps: optim.HostAdamW(ps, lr=1e-3, chunk=7, pipeline=True, slots=2, **ADAMW)
    )
    assert _equal(_bits(direct), _bits(staged))
    for i in d_opt.state:
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(d_opt.state[i][k], s_opt.state[i][k])


def test_promote_keeps_cpu_parameters_host_side_and_checkpoints_host_tensors():
    # covers: RUNTIME-I24
    # WOULD-FAIL-IF: promote moved a CPU parameter's moments (nothing to gain) or reported bytes it did not move,
    # or a checkpoint carried a device tensor where the host layout is promised.
    params, opt = _train(lambda ps: optim.HostAdamW(ps, lr=1e-3, **ADAMW), steps=1)
    assert opt.promote(1 << 40) == 0 and opt.resident == set()
    sd = opt.state_dict()
    assert all(
        st[k].device.type == "cpu" for st in sd["state"].values() for k in ("exp_avg", "exp_avg_sq")
    )
