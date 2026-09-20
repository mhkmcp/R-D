"""SPEC §9.6 ablations and the CIFAR-10-C confounder grid."""

import numpy as np
import pytest

from fidnn.data import cifar10c
from fidnn.detect.features import BLOCK_E, CleanFeatures, FaultFeatures, add_block_e
from fidnn.eval import ablations
from fidnn.taps.features import FEATURE_NAMES

TAPS = ["stem", "stage1", "stage2", "logits"]
NAMES = FEATURE_NAMES + BLOCK_E


def make(kind, n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.abs(rng.normal(1.0 + shift, 0.25, (n, len(TAPS), len(FEATURE_NAMES))))
    return kind(add_block_e(v, FEATURE_NAMES), list(TAPS), list(NAMES))


@pytest.fixture(scope="module")
def data():
    return {"fit": make(CleanFeatures, 200, seed=0), "cal": make(CleanFeatures, 150, seed=1),
            "test": make(CleanFeatures, 150, seed=2), "fault": make(FaultFeatures, 120, 3.0, 3)}


def test_subset_and_zeroed_keep_the_container_type(data):
    s = ablations.subset(data["fault"], ["stem", "logits"])
    assert isinstance(s, FaultFeatures) and s.taps == ["stem", "logits"]
    z = ablations.zeroed(data["fit"], ablations.BLOCKS["B"])
    assert isinstance(z, CleanFeatures)
    assert (z.values[:, :, [NAMES.index(n) for n in ablations.BLOCKS["B"]]] == 0).all()
    assert z.values.shape == data["fit"].values.shape   # width stays comparable


def test_k_sweep_is_greedy_and_uses_no_fault_data(data):
    rows = ablations.greedy_tap_selection(data["fit"], data["cal"], k_max=3)
    assert rows.k.tolist() == [1, 2, 3]
    assert all(len(t) == k for t, k in zip(rows.taps, rows.k))
    for earlier, later in zip(rows.taps, rows.taps[1:]):
        assert set(earlier) < set(later), "forward selection only adds taps"
    assert "tpr" not in rows.columns, "selecting on fault data would violate C2"


def test_tap_position_covers_thirds_and_logits_only(data):
    rows = ablations.tap_position(data["fit"], data["cal"], data["test"], data["fault"])
    assert {"early", "late", "logits_only"} <= set(rows.position)
    assert rows[rows.position == "logits_only"].k.iloc[0] == 1
    assert rows.tpr.between(0, 1).all() and rows.measured_fpr.between(0, 1).all()


def test_block_ablation_covers_every_block_and_the_downgrade(data):
    rows = ablations.block_ablation(data["fit"], data["cal"], data["test"], data["fault"])
    assert set(rows.ablation) == {"full", "drop_A", "drop_B", "drop_C", "drop_D", "drop_E",
                                  "d_dense_downgrade"}
    assert rows.tpr.between(0, 1).all()


def test_training_budget_never_exceeds_clean_fit(data):
    rows = ablations.training_budget(data["fit"], data["cal"], data["test"], data["fault"],
                                     budgets=(50, 100, 200, 6000))
    assert rows.clean_fit_n.tolist() == [50, 100, 200]   # 6000 > len(clean_fit): skipped
    assert rows.tpr.between(0, 1).all()


def test_kernel_sensitivity_compares_three_kernels(data):
    rows = ablations.kernel_sensitivity(data["fit"], data["cal"], data["test"], data["fault"])
    assert rows.kernel.tolist() == ["rbf", "poly", "linear"]
    assert rows.nu.nunique() == 1, "ν is held fixed so the kernel is the only variable"


def test_transfer_truncates_to_the_shared_tap_width(data):
    other_taps = ["b1", "b2", "b3"]
    rng = np.random.default_rng(9)
    v = np.abs(rng.normal(1.0, 0.25, (100, 3, len(FEATURE_NAMES))))
    target = CleanFeatures(add_block_e(v, FEATURE_NAMES), other_taps, list(NAMES))
    target_fault = FaultFeatures(add_block_e(v + 3, FEATURE_NAMES), other_taps, list(NAMES))
    out = ablations.transfer(data["fit"], data["cal"], target, target_fault)
    assert out["shared_taps"] == 3
    assert len(out["source_taps"]) == len(out["target_taps"]) == 3
    assert 0 <= out["tpr"] <= 1


def test_cifar10c_grid_is_fifteen_types_by_five_severities():
    grid = cifar10c.grid()
    assert len(grid) == 75
    assert len({t for _, t, _ in grid}) == 15
    assert {f for f, _, _ in grid} == {"noise", "blur", "weather", "digital"}


def test_cifar10c_family_lookup():
    assert cifar10c.family_of("gaussian_noise") == "noise"
    assert cifar10c.family_of("jpeg_compression") == "digital"
    with pytest.raises(KeyError):
        cifar10c.family_of("not_a_corruption")


def test_cifar10c_load_rejects_unknown_type_and_severity(tmp_path):
    with pytest.raises(KeyError):
        cifar10c.load(tmp_path, "sparkles", 1)
    with pytest.raises(ValueError, match="severity"):
        cifar10c.load(tmp_path, "fog", 9)


def test_cifar10c_load_slices_one_severity(tmp_path):
    d = tmp_path / "CIFAR-10-C"
    d.mkdir()
    np.save(d / "fog.npy", np.arange(50_000 * 3, dtype=np.uint8).reshape(50_000, 1, 1, 3))
    np.save(d / "labels.npy", np.tile(np.arange(10), 5000))
    x, y = cifar10c.load(tmp_path, "fog", 3)
    assert len(x) == len(y) == 10_000
    assert x[0, 0, 0, 0] == np.uint8((2 * 10_000 * 3) % 256)
