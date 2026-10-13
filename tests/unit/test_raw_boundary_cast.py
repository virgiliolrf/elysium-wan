"""The raw codec carries the declared fp32 boundary for a narrower compute dtype."""

from types import SimpleNamespace

import pytest
import torch

from wantrain.runtime.stage import StageRuntime as Stage
from wantrain.transport import framing as fr


def _stage(boundary, compute):
    return SimpleNamespace(
        hf=False,
        spec=SimpleNamespace(
            model=SimpleNamespace(
                boundary_dtype=boundary, compute_dtype=compute, fp8_boundaries=False
            )
        ),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_raw_codec_widens_a_narrow_compute_tensor_to_the_fp32_boundary(dtype):
    # covers: RUNTIME-I6
    # WOULD-FAIL-IF: the raw branch sends the compute dtype as is (tensor_payload raises
    # "unsupported dtype"), or the round trip through the fp32 wire changes one bit.
    compute = "bf16" if dtype == torch.bfloat16 else "fp16"
    st = _stage("fp32", compute)
    x = torch.randn(3, 5, 7).to(dtype)
    codec, payload, shape = Stage._encode_body(st, fr.FWD, x, "raw")
    assert codec == fr.RAW_F32
    wire = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape)
    back = Stage._decode_body(st, SimpleNamespace(tensor=lambda: wire))
    assert back.dtype == dtype
    assert torch.equal(back.view(torch.int16), x.view(torch.int16)), (
        "the fp32 wire round trip moved a bit"
    )


def test_raw_codec_leaves_fp32_and_integer_tensors_alone():
    # covers: RUNTIME-I6
    st = _stage("fp32", "fp32")
    x = torch.randn(4, 4)
    codec, payload, _ = Stage._encode_body(st, fr.FWD, x, "raw")
    assert codec == fr.RAW_F32 and payload == x.numpy().tobytes()
    ids = torch.arange(6, dtype=torch.int64)
    assert Stage._encode_body(st, fr.FWD, ids, "raw")[0] == fr.RAW_I64
