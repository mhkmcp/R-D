import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fidnn.data import cifar10
from fidnn.data.loader import Batches, random_crop_flip

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "artifacts" / "data"
needs_data = pytest.mark.skipif(not (PREPARED / "cifar10.json").exists(),
                                reason="needs `fidnn data prepare`")


def _labels(n_per_class=100):
    return np.repeat(np.arange(10), n_per_class)


def test_splits_are_stratified_disjoint_and_deterministic():
    y = _labels()
    s = cifar10.make_splits(y, seed=0)
    assert set(s) == set(cifar10.SPLITS)
    for name, share in zip(cifar10.SPLITS, (0.8, 0.12, 0.04, 0.04)):
        counts = np.bincount(y[s == name], minlength=10)
        assert (counts == round(share * 100)).all(), name
    assert (s == cifar10.make_splits(y, seed=0)).all()
    assert (s != cifar10.make_splits(y, seed=1)).any()


def test_detector_splits_come_from_the_classifier_hold_out_only():
    y = _labels(500)
    s = cifar10.make_splits(y, seed=0)
    train = set(np.flatnonzero(s == "train"))
    for name in cifar10.DETECTOR_SPLITS:
        assert train.isdisjoint(np.flatnonzero(s == name)), name
    assert len(train) + sum((s == n).sum() for n in cifar10.DETECTOR_SPLITS) == len(y)


def test_collisions_and_label_agreement():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    assert cifar10.collisions(a, b) == 0
    assert cifar10.collisions(a, np.concatenate([b, a[:3]])) == 3
    y = np.arange(20) % 10
    ok = cifar10.label_agreement(a, y, a[::-1], y[::-1], n=20, seed=0)
    assert ok == {"checked": 20, "matched": 20, "mismatched": 0}
    bad = cifar10.label_agreement(a, y, a, (y + 1) % 10, n=20, seed=0)
    assert bad["mismatched"] == 20


def test_channel_stats_are_diagonal_only():
    x = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    x[..., 0] = 255
    mean, std = cifar10.channel_stats(x)
    assert len(mean) == len(std) == 3
    assert mean == [1.0, 0.0, 0.0] and std == [0.0, 0.0, 0.0]


def test_augmentation_shape_and_content():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 3, 32, 32)
    out = random_crop_flip(x, 4, g)
    assert out.shape == x.shape
    flipped_only = random_crop_flip(x, 0, torch.Generator().manual_seed(1))
    assert all(torch.equal(f, o) or torch.equal(f, o.flip(2)) for f, o in zip(flipped_only, x))


def test_batches_normalise_and_cover_every_sample():
    x = np.full((10, 32, 32, 3), 255, dtype=np.uint8)
    y = np.arange(10)
    b = Batches(x, y, mean=[0.5] * 3, std=[0.5] * 3, batch_size=4, shuffle=True)
    seen = torch.cat([yy for _, yy in b])
    assert len(b) == 3 and sorted(seen.tolist()) == list(range(10))
    xb, _ = next(iter(b))
    assert torch.allclose(xb, torch.ones_like(xb))


@needs_data
def test_prepared_data_integrity():
    meta = json.loads((PREPARED / "cifar10.json").read_text())
    assert meta["integrity"]["kaggle_vs_canonical_test_collisions"] == 0
    assert meta["integrity"]["label_agreement"]["mismatched"] == 0
    assert meta["counts"] == {"train": 40000, "clean_fit": 6000, "clean_cal": 2000,
                              "clean_test": 2000}
    assert meta["normalisation"]["fitted_on"] == "train"


@needs_data
def test_normalisation_fitted_on_train_only():
    import pandas as pd
    import yaml

    meta = json.loads((PREPARED / "cifar10.json").read_text())
    root = ROOT / yaml.safe_load((ROOT / "configs/data/cifar10.yaml").read_text())["root"]
    x, _ = cifar10.load_kaggle_train(root)
    split = pd.read_parquet(PREPARED / "cifar10_splits.parquet").split.to_numpy()
    train_mean, _ = cifar10.channel_stats(x[split == "train"])
    all_mean, _ = cifar10.channel_stats(x)
    assert np.allclose(meta["normalisation"]["mean"], train_mean)
    assert not np.allclose(train_mean, all_mean, atol=1e-7)
