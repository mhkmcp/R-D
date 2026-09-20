"""SPEC §11.4: feature-block population, D-dense on synthetic tensors, determinism."""

import pytest
import torch

from fidnn.models.registry import build
from fidnn.taps.features import BLOCK_A, BLOCK_B, BLOCK_C, FEATURE_NAMES, WIDTH, tap_features
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps

D = slice(len(BLOCK_A) + len(BLOCK_B) + len(BLOCK_C), WIDTH)


def test_width():
    assert WIDTH == len(FEATURE_NAMES) == 26


@pytest.mark.parametrize("model_id", ["m2", "m3"])
@pytest.mark.parametrize("tap_set", ["default", "extended"])
def test_d_conv_populated_on_every_tap(model_id, tap_set):
    model = build(model_id).eval()
    with TapMonitor(model, taps(model_id, tap_set)) as mon, torch.no_grad():
        model(torch.randn(8, 3, 32, 32))
    for tap_id, f in mon.outputs.items():
        assert f.shape == (8, WIDTH), tap_id
        assert torch.isfinite(f).all(), tap_id
        assert f[:, D][:, :4].abs().sum() > 0, f"D-conv all zero on {tap_id}"


def test_d_dense_on_synthetic_tensor():
    f = tap_features(torch.randn(16, 128), dense=True)
    assert f.shape == (16, WIDTH)
    d = f[:, D]
    assert (d[:, :4] != 0).any(dim=0).all()
    assert (d[:, 4:] == 0).all()


def test_deterministic():
    x = torch.randn(4, 32, 8, 8)
    assert torch.equal(tap_features(x), tap_features(x.clone()))


def test_nan_and_inf_counted():
    x = torch.randn(2, 4, 3, 3)
    x[0, 0, 0, 0] = float("nan")
    x[1, 1, 1, 1] = float("inf")
    f = tap_features(x)
    nan_i, inf_i = FEATURE_NAMES.index("nan_count"), FEATURE_NAMES.index("inf_count")
    assert f[0, nan_i] == 1 and f[1, inf_i] == 1


def test_energy_gini_bounds():
    gini = FEATURE_NAMES.index("energy_gini")
    uniform = tap_features(torch.ones(1, 8, 4, 4))[0, gini]
    peaked = torch.zeros(1, 8, 4, 4)
    peaked[0, 0] = 1.0
    assert abs(uniform) < 1e-5
    assert tap_features(peaked)[0, gini] > 0.8
