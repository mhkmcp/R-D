"""SPEC §4.3: the four outcome labels, their tracks, and the sweep's guarantees."""

import numpy as np
import pandas as pd
import pytest
import torch

from fidnn.inject import sweep
from fidnn.inject.plan import plan
from fidnn.inject.targets import targets
from fidnn.inject.taxonomy import TRACK, label, track
from fidnn.models.registry import build

EPS = 0.01
CLEAN = np.array([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 6.0], [3.0, 2.0, 1.0]])
Y = np.array([0, 1, 2, 0])


def _labels(fault):
    return label(CLEAN, np.asarray(fault), Y, EPS, num_classes=3)


def test_masked_when_nothing_moves():
    assert (_labels(CLEAN + 1e-6) == "MASKED").all()


def test_degraded_when_the_margin_moves_but_the_prediction_does_not():
    fault = CLEAN.copy()
    fault[:, 1] += 0.5
    assert (_labels(fault) == "DEGRADED").all()


def test_sdc_when_top1_changes():
    fault = CLEAN.copy()
    fault[0] = [0.0, 9.0, 0.0]
    assert _labels(fault)[0] == "SDC"


def test_crash_on_non_finite_output_takes_priority():
    fault = CLEAN.copy()
    fault[2] = [np.nan, np.nan, np.nan]
    out = _labels(fault)
    assert out[2] == "CRASH"
    fault[2] = [np.inf, 0.0, 0.0]
    assert _labels(fault)[2] == "CRASH"


def test_crash_when_accuracy_collapses_to_chance():
    clean = np.array([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 6.0], [1.0, 5.0, 0.0]])
    y = np.array([0, 1, 2, 1])                       # clean is 4/4 correct
    fault = np.tile([[9.0, 0.0, 0.0]], (4, 1))       # collapsed to one class: 1/4 <= 1/3
    assert (label(clean, fault, y, EPS, num_classes=3) == "CRASH").all()


def test_epsilon_is_the_masked_degraded_boundary():
    fault = CLEAN.copy()
    fault[0, 2] += EPS / 2
    assert _labels(fault)[0] == "MASKED"
    fault[0, 2] += EPS
    assert _labels(fault)[0] == "DEGRADED"


def test_tracks_never_merge():
    labels = np.array(["MASKED", "DEGRADED", "SDC", "CRASH"], dtype=object)
    assert list(track(labels)) == ["S", "S", "H", "H"]
    assert set(TRACK.values()) == {"S", "H"}


@pytest.fixture(scope="module")
def smoke_outcomes():
    torch.manual_seed(0)
    model = build("m1").eval()
    grid = plan(targets(model, "m1"), "bf_w", seed=0, reps=1, budgets=(1, 16))
    x = torch.randn(64, 3, 32, 32)
    y = np.random.default_rng(0).integers(0, 2, 64)
    with torch.no_grad():
        clean = model(x).numpy()
    return grid, sweep.run(model, grid, x, y, clean, EPS, 2, probes_per_injection=8,
                           checksum_every=5, log=lambda _: None)


def test_sweep_labels_every_injection_probe_pair(smoke_outcomes):
    grid, out = smoke_outcomes
    assert len(out) == grid.injection_id.nunique() * 8
    assert set(out.label) <= {"MASKED", "DEGRADED", "SDC", "CRASH"}
    assert (out.track == out.label.map(TRACK)).all()
    assert set(out.columns) >= {"injection_id", "probe_index", "logit_shift", "bucket", "stratum",
                                "budget", "flips_applied"}


def test_sweep_raises_if_the_model_does_not_return_to_clean(smoke_outcomes):
    grid, _ = smoke_outcomes
    model = build("m1").eval()
    x, y = torch.randn(8, 3, 32, 32), np.zeros(8, dtype=int)
    with torch.no_grad():
        clean = model(x).numpy()

    def corrupt(*_, **__):  # a restore that silently fails must abort the sweep
        model.conv1.weight.data.add_(1.0)

    one = grid[grid.injection_id == grid.injection_id.iloc[0]]
    import fidnn.inject.sweep as sweep_mod
    original = sweep_mod.injected_flips
    try:
        from contextlib import contextmanager

        @contextmanager
        def broken(model_, flips, mode):
            corrupt()
            yield list(flips)

        sweep_mod.injected_flips = broken
        with pytest.raises(sweep.RestoreError):
            sweep.run(model, one, x, y, clean, EPS, 2, probes_per_injection=4, checksum_every=1,
                      log=lambda _: None)
    finally:
        sweep_mod.injected_flips = original


def test_summary_reports_track_s_share(smoke_outcomes):
    _, out = smoke_outcomes
    s = sweep.summarise(out)
    assert {"MASKED", "DEGRADED", "SDC", "CRASH", "n", "track_s_share"} <= set(s.columns)
    assert np.isclose(s.track_s_share.between(0, 1).all(), True)
    expected = (out.track == "S").mean()
    assert np.isclose(sweep.track_s_share(out), expected)
    assert np.isclose((s.n * s.track_s_share).sum() / s.n.sum(), expected)


def test_probe_selection_is_seeded(smoke_outcomes):
    grid, out = smoke_outcomes
    model = build("m1").eval()
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    y = np.random.default_rng(0).integers(0, 2, 64)
    with torch.no_grad():
        clean = model(x).numpy()
    again = sweep.run(model, grid, x, y, clean, EPS, 2, probes_per_injection=8, checksum_every=5,
                      log=lambda _: None)
    pd.testing.assert_series_equal(out.probe_index, again.probe_index)
