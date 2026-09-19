"""SPEC §11.4: tap budget, hook coverage, hook removal."""

import pytest
import torch

from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import MODELS, build
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


@pytest.mark.parametrize("model_id", MODELS)
def test_tap_budget(model_id):
    assert len(taps(model_id, "default")) >= 5


def test_tap_counts_match_spec():
    assert (len(taps("m2", "default")), len(taps("m2", "extended"))) == (6, 13)
    assert (len(taps("m3", "default")), len(taps("m3", "extended"))) == (7, 10)
    assert [t.tap_id for t in taps("m2", "extended") if t.pre_add] == ["s3b3_pre"]


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("tap_set", ["default", "extended"])
@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_each_tap_fires_once(model_id, tap_set, precision):
    model = build(model_id).eval()
    if precision == "int8":
        model = quantize_ptq(model, [torch.randn(4, 3, 32, 32)])
    with TapMonitor(model, taps(model_id, tap_set), "capture") as mon, torch.no_grad():
        model(torch.randn(3, 3, 32, 32))
    assert set(mon.calls.values()) == {1}
    assert all(not v.is_quantized for v in mon.outputs.values())


def test_remove_detaches_all_hooks():
    model = build("m2").eval()
    mon = TapMonitor(model, taps("m2", "extended"))
    assert any(m._forward_hooks for m in model.modules())
    mon.remove()
    assert not any(m._forward_hooks for m in model.modules())


def test_unknown_tap_path_rejected():
    from fidnn.taps.registry import TapInfo
    with pytest.raises(KeyError):
        TapMonitor(build("m1"), [TapInfo("nope", "no.such.module")])
