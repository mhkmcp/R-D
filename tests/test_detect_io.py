"""Reading fault features back off disk: mode parsing and label alignment (SPEC §7).

The in-memory tests never touch the file naming or the pivot ordering, which is exactly where both
of these defects lived.
"""

import numpy as np
import pandas as pd
import pytest

from fidnn.detect.features import CleanFeatures, FaultFeatures, from_frame
from fidnn.detect.run import _fault_features
from fidnn.taps.features import FEATURE_NAMES

TAPS = ["stem", "stage1", "logits"]
KEY = "mean"          # carries injection_id * 100 + probe_index, so alignment is checkable
OUT_COLS = ["out_msp", "out_entropy", "out_margin", "out_energy"]


def _write_arm(features_dir, faults_dir, mode, seed=0, injections=5, probes=4, rng_seed=0):
    """One arm's files, with probe indices deliberately unsorted (as `rng.choice` leaves them)."""
    rng = np.random.default_rng(rng_seed)
    feat_rows, out_rows = [], []
    for injection in range(injections):
        probe_idx = rng.choice(1000, size=probes, replace=False)   # unsorted on purpose
        for tap in TAPS:
            df = pd.DataFrame({n: np.zeros(probes) for n in FEATURE_NAMES})
            df[KEY] = injection * 100 + probe_idx
            df.insert(0, "tap_id", tap)
            df.insert(0, "probe_index", probe_idx)
            df.insert(0, "injection_id", injection)
            feat_rows.append(df)
        labels = np.where(np.arange(probes) % 2 == 0, "MASKED", "SDC")
        out_rows.append(pd.DataFrame({
            "injection_id": injection, "probe_index": probe_idx,
            "mode": mode, "attacker": "L0", "bucket": "late", "stratum": "sign", "budget": 1,
            "label": labels, "track": np.where(labels == "MASKED", "S", "H"),
            **{c: rng.normal(size=probes) for c in OUT_COLS},
        }))
    stem = f"m2_int8_{mode}_seed{seed}"
    pd.concat(feat_rows, ignore_index=True).to_parquet(
        features_dir / f"{stem}_fault_features.parquet", index=False)
    pd.concat(out_rows, ignore_index=True).to_parquet(
        faults_dir / f"{stem}_outcomes.parquet", index=False)


@pytest.mark.parametrize("mode", ["bf_w", "bf_b", "bf_bn", "rnd_val", "sa0"])
def test_mode_with_underscores_is_parsed_from_the_filename(tmp_path, mode):
    features_dir, faults_dir = tmp_path / "features", tmp_path / "faults"
    features_dir.mkdir(), faults_dir.mkdir()
    _write_arm(features_dir, faults_dir, mode)
    features, labels = _fault_features(features_dir, faults_dir, "m2", "int8", 0, TAPS)
    assert set(labels["mode"]) == {mode}
    assert len(features) == len(labels) == 5 * 4


def test_labels_line_up_with_their_feature_rows(tmp_path):
    """Regression: the pivot sorts its index, the file order does not — they must be aligned."""
    features_dir, faults_dir = tmp_path / "features", tmp_path / "faults"
    features_dir.mkdir(), faults_dir.mkdir()
    _write_arm(features_dir, faults_dir, "bf_w")
    features, labels = _fault_features(features_dir, faults_dir, "m2", "int8", 0, TAPS)

    encoded = features.values[:, 0, features.names.index(KEY)]
    expected = labels.injection_id.to_numpy() * 100 + labels.probe_index.to_numpy()
    np.testing.assert_array_equal(encoded, expected)


def test_several_modes_stay_separate_and_aligned(tmp_path):
    features_dir, faults_dir = tmp_path / "features", tmp_path / "faults"
    features_dir.mkdir(), faults_dir.mkdir()
    _write_arm(features_dir, faults_dir, "bf_w", rng_seed=1)
    _write_arm(features_dir, faults_dir, "bf_b", rng_seed=2)
    features, labels = _fault_features(features_dir, faults_dir, "m2", "int8", 0, TAPS)

    assert set(labels["mode"]) == {"bf_w", "bf_b"}
    assert len(features) == len(labels) == 2 * 5 * 4
    encoded = features.values[:, 0, features.names.index(KEY)]
    expected = labels.injection_id.to_numpy() * 100 + labels.probe_index.to_numpy()
    np.testing.assert_array_equal(encoded, expected)


def test_grid_columns_survive_so_the_splits_and_guard_can_run(tmp_path):
    features_dir, faults_dir = tmp_path / "features", tmp_path / "faults"
    features_dir.mkdir(), faults_dir.mkdir()
    _write_arm(features_dir, faults_dir, "bf_w")
    _, labels = _fault_features(features_dir, faults_dir, "m2", "int8", 0, TAPS)
    assert {"bucket", "stratum", "budget", "attacker", "label", "track"} <= set(labels.columns)
    assert set(OUT_COLS) <= set(labels.columns), "output-only baselines need these at eval"


def test_no_fault_features_returns_none(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "faults").mkdir()
    assert _fault_features(tmp_path / "features", tmp_path / "faults", "m2", "int8", 0, TAPS) is None


def test_from_frame_returns_the_index_it_sorted_by(tmp_path):
    rows = []
    for probe in (7, 3, 11):
        for tap in TAPS:
            df = pd.DataFrame({n: np.zeros(1) for n in FEATURE_NAMES})
            df[KEY] = float(probe)
            df.insert(0, "tap_id", tap)
            df.insert(0, "index", probe)
            rows.append(df)
    features, index = from_frame(pd.concat(rows, ignore_index=True), TAPS, CleanFeatures)
    assert list(index) == [3, 7, 11]
    np.testing.assert_array_equal(features.values[:, 0, features.names.index(KEY)],
                                  [3.0, 7.0, 11.0])
    assert isinstance(features, CleanFeatures) and not isinstance(features, FaultFeatures)
