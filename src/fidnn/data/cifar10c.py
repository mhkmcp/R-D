"""CIFAR-10-C: the §9.6(8) input-shift confounder. Clean inputs only, never mixed with faults."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ZENODO_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
SEVERITIES = (1, 2, 3, 4, 5)
FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
TYPES = tuple(t for ts in FAMILIES.values() for t in ts)   # the 15 mandatory types


def family_of(corruption: str) -> str:
    for family, members in FAMILIES.items():
        if corruption in members:
            return family
    raise KeyError(f"unknown corruption {corruption}")


def download(root: Path) -> Path:
    """Fetch and unpack the Zenodo archive (CC BY 4.0). ~2.9 GB; no credentials needed."""
    root.mkdir(parents=True, exist_ok=True)
    tar = root / "CIFAR-10-C.tar"
    if not (root / "CIFAR-10-C").is_dir():
        if not tar.exists():
            urllib.request.urlretrieve(ZENODO_URL, tar)
        with tarfile.open(tar) as f:
            f.extractall(root, filter="data")
    return root / "CIFAR-10-C"


def load(root: Path, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """The 10,000 images of one (type, severity), with the shared labels."""
    if corruption not in TYPES:
        raise KeyError(f"{corruption} is not one of the 15 mandatory types")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    d = root / "CIFAR-10-C"
    x = np.load(d / f"{corruption}.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    lo = (severity - 1) * 10_000
    return np.asarray(x[lo:lo + 10_000]), np.asarray(y[lo:lo + 10_000])


def grid() -> list[tuple[str, str, int]]:
    """(family, type, severity) — the per-type and per-family breakdowns are mandatory (§9.6)."""
    return [(family_of(t), t, s) for t in TYPES for s in SEVERITIES]
