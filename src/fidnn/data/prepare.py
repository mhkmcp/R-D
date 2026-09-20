"""`fidnn data prepare`: integrity checks, persisted splits, train channel stats (SPEC §3.2)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fidnn.data import cifar10
from fidnn.provenance import sidecar


class IntegrityError(RuntimeError):
    pass


def prepare(config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    root = Path(cfg["root"])
    kx, ky = cifar10.load_kaggle_train(root)
    tx, ty = cifar10.load_canonical(root, train=False)
    cx, cy = cifar10.load_canonical(root, train=True)

    if kx.shape != (50_000, 32, 32, 3) or np.bincount(ky, minlength=10).tolist() != [5000] * 10:
        raise IntegrityError(f"Kaggle train: shape {kx.shape}, class counts {np.bincount(ky)}")
    if tx.shape != (10_000, 32, 32, 3) or np.bincount(ty, minlength=10).tolist() != [1000] * 10:
        raise IntegrityError(f"canonical test: shape {tx.shape}, class counts {np.bincount(ty)}")

    n_collisions = cifar10.collisions(kx, tx)
    agreement = cifar10.label_agreement(kx, ky, cx, cy, cfg["label_check_n"], cfg["split_seed"])
    if n_collisions:
        raise IntegrityError(f"{n_collisions} Kaggle-train images appear in the canonical test set; "
                             "SPEC §3.2 fallback: use the canonical distribution for both splits")
    if agreement["mismatched"] or agreement["matched"] < 0.99 * agreement["checked"]:
        raise IntegrityError(f"label agreement failed: {agreement}")

    splits = cifar10.make_splits(ky, cfg["split_seed"], cfg["train_fraction"],
                                 tuple(cfg["fractions"]))
    mean, std = cifar10.channel_stats(kx[splits == "train"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(len(ky)), "label": ky, "split": splits}).to_parquet(
        out_dir / "cifar10_splits.parquet", index=False)
    record = {
        "normalisation": {"mean": mean, "std": std, "fitted_on": "train"},
        "counts": {s: int((splits == s).sum()) for s in cifar10.SPLITS},
        "probe_pool": {"source": "torchvision CIFAR10(train=False)", "n": len(ty)},
        "integrity": {"kaggle_vs_canonical_test_collisions": n_collisions,
                      "label_agreement": agreement},
        "sha256": {name: cifar10.sha256(root / "kaggle" / name) for name in cifar10.KAGGLE_FILES}
        | {"cifar-10-python.tar.gz": cifar10.sha256(root / "canonical" / "cifar-10-python.tar.gz")},
        **sidecar(cfg, seed=cfg["split_seed"], device="cpu"),
    }
    (out_dir / "cifar10.json").write_text(json.dumps(record, indent=2))
    return record


def load_prepared(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    return (pd.read_parquet(out_dir / "cifar10_splits.parquet"),
            json.loads((out_dir / "cifar10.json").read_text()))


def subset_classes(x: np.ndarray, y: np.ndarray, classes: list[int] | None):
    """Keep only `classes` and remap their labels to 0..n-1 (M1's 2-class subset, SPEC §3)."""
    if not classes:
        return x, y
    keep = np.isin(y, classes)
    remap = {c: i for i, c in enumerate(classes)}
    return x[keep], np.vectorize(remap.get)(y[keep]).astype(np.int64)


def model_classes(model_cfg: Path) -> list[int] | None:
    """The class subset a model is trained on, or None for the full 10 classes."""
    return yaml.safe_load(model_cfg.read_text()).get("classes")


@dataclass(frozen=True)
class Arrays:
    """Prepared images by split, plus the frozen normalisation (SPEC §3.2)."""

    kaggle_x: np.ndarray
    kaggle_y: np.ndarray
    split: np.ndarray
    probe_x: np.ndarray       # canonical test set: the labelled fault probe pool
    probe_y: np.ndarray
    mean: list[float]
    std: list[float]

    def of(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        keep = self.split == name
        return self.kaggle_x[keep], self.kaggle_y[keep]


def load_arrays(config_path: Path, out_dir: Path) -> Arrays:
    root = Path(yaml.safe_load(config_path.read_text())["root"])
    splits, meta = load_prepared(out_dir)
    kx, ky = cifar10.load_kaggle_train(root)
    px, py = cifar10.load_canonical(root, train=False)
    return Arrays(kx, ky, splits.split.to_numpy(), px, py,
                  meta["normalisation"]["mean"], meta["normalisation"]["std"])
