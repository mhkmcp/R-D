"""Two-source integrity checks: source disjointness and label agreement (SPEC §3.2, Dataset.md §5.1)."""

import hashlib

import numpy as np


def pixel_hashes(x: np.ndarray) -> list[bytes]:
    """One digest per image over its raw uint8 pixels; exact duplicates only (Dataset.md §5.2)."""
    x = np.ascontiguousarray(x, dtype=np.uint8)
    return [hashlib.blake2b(img.tobytes(), digest_size=16).digest() for img in x]


def disjointness(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel-hash collisions between two image sets. `collisions` must be 0."""
    ha, hb = pixel_hashes(a), set(pixel_hashes(b))
    hits = [i for i, h in enumerate(ha) if h in hb]
    return {"n_a": len(ha), "n_b": len(hb), "collisions": len(hits), "a_indices": hits[:20]}


def label_agreement(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> dict:
    """Match each image in `x` to `ref_x` by pixel hash and compare labels.

    `unmatched` counts images with no exact twin in the reference; `ambiguous` counts reference
    hashes shared by images of different labels (CIFAR-10 has a few exact duplicates).
    """
    ref: dict[bytes, set[int]] = {}
    for h, label in zip(pixel_hashes(ref_x), ref_y.tolist()):
        ref.setdefault(h, set()).add(label)
    agree = disagree = unmatched = ambiguous = 0
    mismatches = []
    for i, (h, label) in enumerate(zip(pixel_hashes(x), y.tolist())):
        labels = ref.get(h)
        if labels is None:
            unmatched += 1
        elif len(labels) > 1:
            ambiguous += 1
        elif label in labels:
            agree += 1
        else:
            disagree += 1
            mismatches.append(i)
    return {"checked": len(y), "agree": agree, "disagree": disagree, "unmatched": unmatched,
            "ambiguous": ambiguous, "mismatch_indices": mismatches[:20]}
