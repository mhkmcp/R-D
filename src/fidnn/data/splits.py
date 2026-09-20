"""Stratified, seeded, persisted splits of the Kaggle 50k (SPEC §3.2, §7)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# per-class counts out of 5,000: classifier `train` 80 %, then 60/20/20 of the held-out 1,000
PER_CLASS = {"train": 4000, "clean_fit": 600, "clean_cal": 200, "clean_test": 200}
DETECTOR_SPLITS = ("clean_fit", "clean_cal", "clean_test")
SPLIT_SEED = 0  # one partition for the whole project; model seeds do not re-split


def make_splits(y: np.ndarray, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Row indices per split; every class contributes exactly `PER_CLASS` rows to each."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {k: [] for k in PER_CLASS}
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        if len(idx) != sum(PER_CLASS.values()):
            raise ValueError(f"class {c} has {len(idx)} images, expected {sum(PER_CLASS.values())}")
        start = 0
        for name, n in PER_CLASS.items():
            out[name].append(idx[start:start + n])
            start += n
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def save_splits(splits: dict[str, np.ndarray], ids: np.ndarray, y: np.ndarray, path: Path,
                seed: int) -> str:
    rows = [{"row": int(i), "kaggle_id": int(ids[i]), "label": int(y[i]), "split": name}
            for name, idx in splits.items() for i in idx]
    df = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(
        {"seed": seed, "per_class": PER_CLASS, "content_sha256": digest}, indent=2))
    return digest


def load_splits(path: Path) -> dict[str, np.ndarray]:
    """Always reload the persisted partition rather than re-deriving it."""
    df = pd.read_parquet(path)
    return {name: np.sort(df.loc[df.split == name, "row"].to_numpy()) for name in PER_CLASS}
