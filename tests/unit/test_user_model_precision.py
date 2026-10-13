"""compute_dtype must mean the same thing for an operator's model as for ours:
built-in blocks open their own autocast region, a user model has none, and the
engine promotes every arriving activation to the compute dtype, so a mid-stage
user model got bf16 inputs against fp32 parameters. The engine supplies it."""

# WOULD-FAIL-IF: a user-defined model stops getting the engine's autocast region (narrow activations meet fp32 weights), or fp32 user models are altered

import contextlib
import os
import platform
import random
import subprocess
import sys
import textwrap

import pytest
import torch

from wantrain.model.gpt import GPTConfig, build_layers
from wantrain.runtime import determinism
from wantrain.runtime.determinism import DeterminismError, check_determinism, setup_determinism
from wantrain.runtime.stage import compute_autocast, engine_supplies_autocast

# The failure shape: q and k are cast to the input's dtype and v is not.
USER_MODEL = textwrap.dedent("""
    import torch, torch.nn as nn, torch.nn.functional as F

    class Embed(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        def forward(self, idx):
            return self.tok(idx)

    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.h, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
            self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        def forward(self, x):
            B, T, D = x.shape
            q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
            k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2)
            v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
            q, k = q.to(x.dtype), k.to(x.dtype)      # and v is left alone
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return x + self.o(a.transpose(1, 2).reshape(B, T, D))

    class Head(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        def forward(self, x):
            return self.out(x)

    def build(cfg):
        return nn.ModuleList([Embed(cfg)] + [Block(cfg) for _ in range(2)]
                             + [Head(cfg)])
""")


def _cfg(tmp_path, compute="bf16"):
    src = tmp_path / "user_model.py"
    src.write_text(USER_MODEL)
    return GPTConfig(
        arch="user",
        user_model=str(src),
        n_layer=2,
        n_head=4,
        d_model=32,
        seq_len=8,
        vocab_size=64,
        compute_dtype=compute,
        boundary_dtype=compute,
    )


def _mid_stage_input(cfg):
    """What a non-first stage receives: _decode_body promotes every arriving
    activation to the compute dtype, ready or not."""
    return torch.randn(2, cfg.seq_len, cfg.d_model, dtype=cfg.compute_torch_dtype())


# ---- the decision -----------------------------------------------------------


def test_engine_supplies_autocast_only_for_user_models(tmp_path):
    # covers: RUNTIME-I6
    assert engine_supplies_autocast(_cfg(tmp_path, "bf16"))
    # built-ins autocast inside their own forward; touching them moves pinned baselines
    assert not engine_supplies_autocast(
        GPTConfig(n_layer=2, n_head=4, d_model=32, seq_len=8, vocab_size=64, compute_dtype="bf16")
    )
    # fp32 means there is no region to open for anybody.
    assert not engine_supplies_autocast(_cfg(tmp_path, "fp32"))


def test_not_needed_is_a_real_nullcontext(tmp_path):
    """Not-needed must be a nullcontext; autocast(enabled=False) would switch it off."""
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "fp32")
    assert isinstance(compute_autocast(cfg, "cpu"), contextlib.nullcontext)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with compute_autocast(cfg, "cpu"):
            assert torch.is_autocast_enabled("cpu"), (
                "a not-needed region must not switch autocast off"
            )


# ---- the behaviour ----------------------------------------------------------


def test_user_model_crashes_on_a_narrow_input_without_the_region(tmp_path):
    """Pin the bug itself, so a regression reproduces rather than resurfaces."""
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "bf16")
    layers = build_layers(cfg, seed=0)
    x = _mid_stage_input(cfg)
    with pytest.raises(RuntimeError, match="same dtype"):
        out = x
        for layer in layers[1:]:  # skip Embed: mid-stage, as rank 1 is
            out = layer(out)


def test_the_region_makes_the_same_model_run(tmp_path):
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "bf16")
    layers = build_layers(cfg, seed=0)
    x = _mid_stage_input(cfg)
    with compute_autocast(cfg, "cpu"):
        out = x
        for layer in layers[1:]:
            out = layer(out)
    assert torch.isfinite(out.float()).all()


def test_the_matmuls_actually_run_narrow(tmp_path):
    """The matmuls must actually run narrow, not merely stop crashing."""
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "bf16")
    layers = build_layers(cfg, seed=0)
    with compute_autocast(cfg, "cpu"):
        seen = layers[1].q(torch.randn(2, cfg.seq_len, cfg.d_model))
    assert seen.dtype == torch.bfloat16, seen.dtype
    # master weights stay fp32 for the optimizer: autocast casts, it does not overwrite
    assert layers[1].q.weight.dtype == torch.float32


def test_recompute_reproduces_the_forward(tmp_path):
    """_rebuild must run in the same region as _forward, or recompute is not exact."""
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "bf16")
    layers = build_layers(cfg, seed=0)
    x = _mid_stage_input(cfg)

    def run():
        with compute_autocast(cfg, "cpu"):
            out = x
            for layer in layers[1:]:
                out = layer(out)
            return out

    assert torch.equal(run(), run())


def test_fp32_user_models_are_left_exactly_as_they_were(tmp_path):
    """fp32 user models must be left exactly as they were, bit for bit."""
    # covers: RUNTIME-I6
    cfg = _cfg(tmp_path, "fp32")
    layers = build_layers(cfg, seed=0)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    out = x
    for layer in layers[1:]:
        out = layer(out)
    with compute_autocast(cfg, "cpu"):
        guarded = x
        for layer in layers[1:]:
            guarded = layer(guarded)
    assert out.dtype == torch.float32
    assert torch.equal(out, guarded)


# ---- determinism pins (RUNTIME-I19, HANDOUT-M4 I-8) -------------------------

_PIN_VARS = ("CUBLAS_WORKSPACE_CONFIG", "ATEN_CPU_CAPABILITY", "MKL_ENABLE_INSTRUCTIONS")


@pytest.fixture
def restore_determinism(monkeypatch):
    """Every test leaves os.environ, the torch flags and the RNGs as it found them."""
    env = dict(os.environ)
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    py_rng, torch_rng = random.getstate(), torch.get_rng_state()
    for name in _PIN_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(env)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
    (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    ) = cudnn
    random.setstate(py_rng)
    torch.set_rng_state(torch_rng)


@pytest.fixture
def x86(restore_determinism, monkeypatch):
    """A simulated x86 host whose ATen honoured the AVX2 pin."""
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "AVX2")


def _old_setup_determinism(seed):
    """The pre-M4 body, kept as a deliberately wrong implementation: setdefault keeps
    any pre-set pin, and the TF32 flags were set only when CUDA was present."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686"):
        os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _preset_isa_refused(setup) -> bool:
    """True when `setup` refuses a pre-set ATEN_CPU_CAPABILITY=avx512, naming the variable,
    the pre-set value and the pin, before any torch flag changes."""
    os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
    torch.use_deterministic_algorithms(False)
    try:
        setup(0)
    except DeterminismError as e:
        msg = str(e)
        return (
            all(s in msg for s in ("ATEN_CPU_CAPABILITY", "avx512", "avx2"))
            and not torch.are_deterministic_algorithms_enabled()
        )
    return False


def _tf32_off_after(setup) -> bool:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    setup(0)
    return not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def test_determinism_preset_isa_pin_is_refused(x86):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a pre-set ATEN_CPU_CAPABILITY survives (setdefault), or the refusal
    # comes after a torch flag changed
    assert _preset_isa_refused(setup_determinism)
    assert issubclass(DeterminismError, ValueError)


def test_determinism_allow_tf32_is_false_after_setup(x86):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: allow_tf32 is True after setup (set only when CUDA is present)
    assert _tf32_off_after(setup_determinism)
    assert torch.get_float32_matmul_precision() == "highest"


def test_determinism_old_setdefault_body_fails_the_first_two(x86):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the two checks above cannot tell the old setdefault body from the fix
    assert not _preset_isa_refused(_old_setup_determinism)
    os.environ.pop("ATEN_CPU_CAPABILITY")
    if not torch.cuda.is_available():
        assert not _tf32_off_after(_old_setup_determinism)


def test_determinism_assigns_every_pin_and_flag(x86):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a pin is left unassigned, a pre-set equal pin is refused, or a flag is not set
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # equal to the pin: passes
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    setup_determinism(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
    assert os.environ["MKL_ENABLE_INSTRUCTIONS"] == "AVX2"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    check_determinism()


def test_determinism_launchers_export_the_pins_before_the_child_starts(x86, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a stage child starts without the ISA pins in its environment (ATen and
    # MKL would choose their dispatch at the first kernel, before setup_determinism assigns
    # them), or a parent's differing pin is overwritten instead of reaching the child's refusal
    from wantrain.launch import local
    from wantrain.runtime.determinism import pinned_env
    from wantrain.worker.agent import WorkerAgent

    for name in ("CUBLAS_WORKSPACE_CONFIG", "ATEN_CPU_CAPABILITY", "MKL_ENABLE_INSTRUCTIONS"):
        monkeypatch.delenv(name, raising=False)
    want = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    }
    agent = WorkerAgent.__new__(WorkerAgent)
    agent.key_path = "/k"
    for env in (pinned_env({}), local._env(), agent._env()):
        assert {k: env.get(k) for k in want} == want
    assert pinned_env({"ATEN_CPU_CAPABILITY": "avx512"})["ATEN_CPU_CAPABILITY"] == "avx512"
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert "ATEN_CPU_CAPABILITY" not in pinned_env({})


def test_determinism_alloc_conf_stays_a_default(x86):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the allocator setting (which moves no bit) becomes a pin and refuses the operator's value
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    setup_determinism(0)
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


@pytest.mark.parametrize(
    "name,value",
    [("CUBLAS_WORKSPACE_CONFIG", ":16:8"), ("MKL_ENABLE_INSTRUCTIONS", "AVX512")],
)
def test_determinism_preset_differing_pin_is_named(x86, name, value):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: any pre-set differing pin survives setup
    os.environ[name] = value
    with pytest.raises(DeterminismError, match=rf"{name}.*{value}.*{determinism._pins()[name]}"):
        setup_determinism(0)
    assert os.environ[name] == value


_WRONG_FLAGS = [
    ("CUBLAS_WORKSPACE_CONFIG", lambda: os.environ.__setitem__("CUBLAS_WORKSPACE_CONFIG", ":16:8")),
    ("ATEN_CPU_CAPABILITY", lambda: os.environ.__setitem__("ATEN_CPU_CAPABILITY", "avx512")),
    ("MKL_ENABLE_INSTRUCTIONS", lambda: os.environ.pop("MKL_ENABLE_INSTRUCTIONS")),
    ("get_num_threads", lambda: torch.set_num_threads(2)),
    ("are_deterministic_algorithms_enabled", lambda: torch.use_deterministic_algorithms(False)),
    (
        "warn_only",
        lambda: torch.use_deterministic_algorithms(True, warn_only=True),
    ),
    ("float32_matmul_precision", lambda: torch.set_float32_matmul_precision("high")),
    ("cuda.matmul.allow_tf32", lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True)),
    ("cudnn.allow_tf32", lambda: setattr(torch.backends.cudnn, "allow_tf32", True)),
    ("cudnn.deterministic", lambda: setattr(torch.backends.cudnn, "deterministic", False)),
    ("cudnn.benchmark", lambda: setattr(torch.backends.cudnn, "benchmark", True)),
]


@pytest.mark.parametrize("name,force", _WRONG_FLAGS, ids=[n for n, _ in _WRONG_FLAGS])
def test_determinism_readback_names_each_wrong_flag(x86, name, force):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: check_determinism stops reading back a flag or pin, or does not name it
    setup_determinism(0)
    check_determinism()
    force()
    with pytest.raises(DeterminismError, match=name.replace(".", r"\.")):
        check_determinism()


def test_determinism_isa_readback_refuses_a_cached_avx512(restore_determinism, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the ISA pin is assigned after ATen cached the capability and nobody reads it back
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "AVX512")
    with pytest.raises(DeterminismError, match="ATEN_CPU_CAPABILITY.*AVX512"):
        setup_determinism(0)
    assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"  # the pin is in place; ATen ignored it
    with pytest.raises(DeterminismError, match="ATEN_CPU_CAPABILITY.*AVX512"):
        check_determinism()


def test_determinism_isa_not_required_on_arm64(restore_determinism, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an arm64 host is asked for an x86 capability or given the x86 pins
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "DEFAULT")
    setup_determinism(0)
    check_determinism()
    assert "ATEN_CPU_CAPABILITY" not in os.environ
    assert "MKL_ENABLE_INSTRUCTIONS" not in os.environ


def test_determinism_clean_subprocess_sets_no_differing_pin():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the ladder's environment pre-sets a differing pin, or setup refuses a clean process
    code = "from wantrain.runtime.determinism import setup_determinism; setup_determinism(0)"
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr


def test_determinism_import_pins_the_isa_before_torch_is_imported():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: determinism.py imports torch (or anything that dispatches a kernel)
    # before it puts the x86 pins in its own process's environment, so an in-process
    # StageRuntime on an AVX512 host reads ATen's cached AVX512 dispatch and is refused
    code = textwrap.dedent(
        """
        import os, platform, sys
        platform.machine = lambda: "x86_64"
        seen = []
        class Spy:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" and not seen:
                    seen.append((os.environ.get("ATEN_CPU_CAPABILITY"),
                                 os.environ.get("MKL_ENABLE_INSTRUCTIONS")))
        sys.meta_path.insert(0, Spy())
        import wantrain.runtime.determinism
        print(seen)
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in _PIN_VARS}
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[('avx2', 'AVX2')]"
    # a differing pre-set pin is kept for setup_determinism to refuse, never overwritten
    code = "import os, wantrain.runtime.determinism; print(os.environ['ATEN_CPU_CAPABILITY'])"
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(env, ATEN_CPU_CAPABILITY="avx512"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "avx512"


def test_determinism_test_session_holds_the_pins(pytestconfig):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: pyproject's addopts loses `-p wantrain.runtime.determinism`, so a test
    # module that runs a kernel before importing wantrain leaves an x86 session's ATen on
    # its own (AVX512) dispatch and every in-process StageRuntime is refused
    assert pytestconfig.pluginmanager.has_plugin("wantrain.runtime.determinism")
    for name, pin in determinism._pins().items():
        assert os.environ.get(name) == pin, name
    if determinism._is_x86():
        assert torch.backends.cpu.get_cpu_capability().lower() == "avx2"
