"""SPEC §5.2–§5.3, §11.4: extraction shape, determinism, saturation fit, fault-side capture."""

import numpy as np
import pandas as pd
import pytest
import torch

from fidnn.inject.plan import plan
from fidnn.inject.sweep import run as sweep_run
from fidnn.inject.targets import targets
from fidnn.models.registry import build
from fidnn.taps.extract import features, saturation_thresholds
from fidnn.taps.features import BLOCK_D, FEATURE_NAMES
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build("m2").eval()


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(24, 3, 32, 32)


@pytest.mark.parametrize("tap_set", ["default", "extended"])
def test_features_cover_every_sample_and_tap(model, x, tap_set):
    tap_list = taps("m2", tap_set)
    df = features(model, tap_list, x, sat={}, batch=8)
    assert len(df) == len(x) * len(tap_list)
    assert set(df.tap_id) == {t.tap_id for t in tap_list}
    assert list(df.columns) == ["index", "tap_id", *FEATURE_NAMES]
    assert df.groupby("tap_id").size().eq(len(x)).all()
    assert sorted(df[df.tap_id == tap_list[0].tap_id]["index"]) == list(range(len(x)))


def test_extraction_is_deterministic(model, x):
    tap_list = taps("m2", "default")
    a = features(model, tap_list, x, sat={}, batch=8)
    b = features(model, tap_list, x, sat={}, batch=8)
    pd.testing.assert_frame_equal(a, b)


def test_batch_size_changes_features_only_in_the_last_bits(model, x):
    """Conv reductions are not associative, so batching perturbs ~1e-4 relative — no more.

    Determinism (§11.4) is per batching; the extraction batch size goes in the sidecar.
    """
    tap_list = taps("m2", "default")
    a = features(model, tap_list, x, sat={}, batch=24).sort_values(["tap_id", "index"])
    b = features(model, tap_list, x, sat={}, batch=5).sort_values(["tap_id", "index"])
    np.testing.assert_allclose(a[FEATURE_NAMES].to_numpy(), b[FEATURE_NAMES].to_numpy(),
                               rtol=1e-3, atol=1e-4)


def test_block_d_is_populated_on_every_conv_tap(model, x):
    df = features(model, taps("m2", "extended"), x, sat={}, batch=8)
    for tap_id, g in df.groupby("tap_id"):
        if tap_id == "logits":
            continue
        assert g[BLOCK_D].abs().to_numpy().sum() > 0, tap_id


def test_saturation_thresholds_come_from_the_given_clean_sample(model, x):
    tap_list = taps("m2", "default")
    sat = saturation_thresholds(model, tap_list, x, quantile=0.999)
    assert set(sat) == {t.tap_id for t in tap_list}
    assert all(np.isfinite(v) for v in sat.values())

    with_sat = features(model, tap_list, x, sat=sat, batch=8)
    without = features(model, tap_list, x, sat={}, batch=8)
    assert with_sat.sat_frac.sum() > 0          # a finite threshold saturates something
    assert without.sat_frac.sum() == 0          # the default inf threshold saturates nothing


def test_saturation_threshold_rises_with_the_quantile(model, x):
    tap_list = taps("m2", "default")
    low = saturation_thresholds(model, tap_list, x, quantile=0.5)
    high = saturation_thresholds(model, tap_list, x, quantile=0.999)
    assert all(high[k] >= low[k] for k in low)


def test_sweep_records_fault_features_per_injection_and_probe(model):
    torch.manual_seed(2)
    probes = torch.randn(16, 3, 32, 32)
    y = np.random.default_rng(0).integers(0, 10, 16)
    with torch.no_grad():
        clean = model(probes).numpy()
    grid = plan(targets(model, "m2"), "bf_w", seed=0, reps=1, budgets=(1,))
    tap_list = taps("m2", "default")
    collected: list[pd.DataFrame] = []
    with TapMonitor(model, tap_list, mode="features") as mon:
        out = sweep_run(model, grid, probes, y, clean, 0.01, 10, probes_per_injection=4,
                        checksum_every=100, log=lambda _: None, monitor=mon,
                        features_out=collected)
    feats = pd.concat(collected, ignore_index=True)
    n_injections = grid.injection_id.nunique()
    assert len(feats) == n_injections * 4 * len(tap_list)
    assert set(feats.columns) == {"injection_id", "probe_index", "tap_id", *FEATURE_NAMES}
    assert set(feats.injection_id) == set(out.injection_id)
    pairs = feats[["injection_id", "probe_index"]].drop_duplicates()
    assert len(pairs) == n_injections * 4
