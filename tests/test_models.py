"""Model lineup (SPEC §3) and INT8 conversion."""

import pytest
import torch

from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import MODELS, build


def test_resnet20_parameter_count():
    n = sum(p.numel() for p in build("m2").parameters())
    assert abs(n - 270_000) / 270_000 < 0.02


@pytest.mark.parametrize("model_id", MODELS)
def test_output_shape(model_id):
    out = build(model_id).eval()(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, MODELS[model_id].num_classes)


@pytest.mark.parametrize("model_id", MODELS)
def test_int8_keeps_tap_modules_and_does_not_mutate_input(model_id):
    fp32 = build(model_id).eval()
    before = {k: v.clone() for k, v in fp32.state_dict().items()}
    q = quantize_ptq(fp32, [torch.randn(4, 3, 32, 32)])
    fp32_taps = {n for n, _ in fp32.named_modules() if n.rsplit(".", 1)[-1].startswith("tap")}
    q_taps = {n for n, _ in q.named_modules() if n.rsplit(".", 1)[-1].startswith("tap")}
    assert fp32_taps and fp32_taps == q_taps
    assert all(torch.equal(before[k], v) for k, v in fp32.state_dict().items())
    assert q(torch.randn(2, 3, 32, 32)).shape == (2, MODELS[model_id].num_classes)
