"""CIFAR-10 two-source assembly: Kaggle train + canonical test (SPEC §3.2)."""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
KAGGLE_FILES = ("train.7z", "trainLabels.csv")  # never test.7z (SPEC §3.2)
DETECTOR_SPLITS = ("clean_fit", "clean_cal", "clean_test")
SPLITS = ("train", *DETECTOR_SPLITS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_kaggle(root: Path) -> None:
    """Fetch `KAGGLE_FILES` one by one with the kaggle CLI; auth is the user's."""
    kaggle = Path(sys.executable).with_name("kaggle")
    dest = root / "kaggle"
    dest.mkdir(parents=True, exist_ok=True)
    for name in KAGGLE_FILES:
        if (dest / name).exists():
            continue
        subprocess.run([str(kaggle), "competitions", "download", "cifar-10", "-f", name,
                        "-p", str(dest)], check=True)
        wrapped = dest / f"{name}.zip"
        if wrapped.exists():
            with zipfile.ZipFile(wrapped) as z:
                z.extractall(dest)
            wrapped.unlink()


def load_kaggle_train(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """(50000, 32, 32, 3) uint8 images ordered by id, int64 labels. Cached as npz."""
    dest = root / "kaggle"
    cache = dest / "train.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]

    import py7zr
    from PIL import Image

    if not (dest / "train").is_dir():
        with py7zr.SevenZipFile(dest / "train.7z", "r") as archive:
            archive.extractall(path=dest)
    labels = pd.read_csv(dest / "trainLabels.csv").sort_values("id")
    x = np.stack([np.asarray(Image.open(dest / "train" / f"{i}.png").convert("RGB"))
                  for i in labels.id])
    y = labels.label.map(CLASSES.index).to_numpy(np.int64)
    np.savez(cache, x=x, y=y)
    shutil.rmtree(dest / "train")
    return x, y


def load_canonical(root: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(root / "canonical"), train=train, download=True)
    return ds.data, np.asarray(ds.targets, dtype=np.int64)


def pixel_hashes(x: np.ndarray) -> list[str]:
    return [hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest() for img in x]


def collisions(a: np.ndarray, b: np.ndarray) -> int:
    return len(set(pixel_hashes(a)) & set(pixel_hashes(b)))


def label_agreement(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
                    n: int, seed: int) -> dict:
    """Compare labels of `n` sampled images against the reference set, matched by pixel hash."""
    ref = {}
    for h, label in zip(pixel_hashes(ref_x), ref_y):
        ref.setdefault(h, set()).add(int(label))
    idx = np.random.default_rng(seed).choice(len(x), size=min(n, len(x)), replace=False)
    hashes = pixel_hashes(x[idx])
    matched = [(ref[h], int(y[i])) for h, i in zip(hashes, idx) if h in ref]
    return {"checked": len(idx), "matched": len(matched),
            "mismatched": sum(label not in labels for labels, label in matched)}


def make_splits(y: np.ndarray, seed: int, train_fraction: float = 0.8,
                fractions=(0.6, 0.2, 0.2)) -> np.ndarray:
    """Stratified split labels (`SPLITS`) for every index of `y`. See SPEC §3.2."""
    idx = np.arange(len(y))
    train, held = train_test_split(idx, train_size=train_fraction, stratify=y, random_state=seed)
    fit, rest = train_test_split(held, train_size=fractions[0], stratify=y[held],
                                 random_state=seed)
    cal_share = fractions[1] / (fractions[1] + fractions[2])
    cal, test = train_test_split(rest, train_size=cal_share, stratify=y[rest], random_state=seed)
    out = np.empty(len(y), dtype=object)
    for name, part in zip(SPLITS, (train, fit, cal, test)):
        out[part] = name
    return out


def channel_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-channel mean/std of uint8 NHWC images in [0, 1] — diagonal scaling only (C1)."""
    f = x.reshape(-1, 3).astype(np.float64) / 255.0
    return f.mean(0).tolist(), f.std(0).tolist()
