"""Feature containers and Block E fusion (SPEC §5.2, §7). C2 is enforced by the type."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fidnn.taps.features import FEATURE_NAMES

BLOCK_E = ["e_l2_ratio", "e_entropy_ratio", "e_sat_diff"]
WIDTH = len(FEATURE_NAMES) + len(BLOCK_E)  # 29 per tap (SPEC §5.2)
_EPS = 1e-12


@dataclass(frozen=True)
class Features:
    """(N, K, 29) per-sample, per-tap features in registry tap order."""

    values: np.ndarray
    taps: list[str]
    names: list[str]

    def __post_init__(self):
        if self.values.shape[1:] != (len(self.taps), len(self.names)):
            raise ValueError(f"values {self.values.shape} do not match "
                             f"{len(self.taps)} taps × {len(self.names)} features")

    def __len__(self) -> int:
        return len(self.values)

    def tap(self, tap_id: str) -> np.ndarray:
        return self.values[:, self.taps.index(tap_id), :]

    def fused(self) -> np.ndarray:
        """Early-fusion matrix (N, K·29) — D2's input."""
        return self.values.reshape(len(self.values), -1)


class CleanFeatures(Features):
    """Features from clean inferences: the only thing any fit or calibration may see (C2)."""


class FaultFeatures(Features):
    """Features from injected inferences: scoring only, never fitting (C2)."""


def add_block_e(values: np.ndarray, names: list[str]) -> np.ndarray:
    """Cross-tap ratios against the previous tap; the first tap gets neutral values (§5.2)."""
    idx = {n: i for i, n in enumerate(names)}
    l2, ent, sat = values[..., idx["l2_rms"]], values[..., idx["energy_entropy"]], \
        values[..., idx["sat_frac"]]
    prev_l2, prev_ent, prev_sat = (np.roll(a, 1, axis=1) for a in (l2, ent, sat))
    prev_l2[:, 0], prev_ent[:, 0], prev_sat[:, 0] = l2[:, 0], ent[:, 0], sat[:, 0]
    block_e = np.stack([l2 / (prev_l2 + _EPS), ent / (prev_ent + _EPS), sat - prev_sat], axis=-1)
    return np.concatenate([values, block_e], axis=-1)


def from_frame(df: pd.DataFrame, taps: list[str], kind: type[Features],
               index_cols: tuple[str, ...] = ("index",)) -> Features:
    """Long extraction output → (N, K, 29), tap order fixed by the registry."""
    missing = set(taps) - set(df.tap_id.unique())
    if missing:
        raise ValueError(f"features are missing taps {sorted(missing)}")
    wide = df.pivot_table(index=list(index_cols), columns="tap_id", values=FEATURE_NAMES)
    values = np.stack([np.stack([wide[(f, t)].to_numpy() for f in FEATURE_NAMES], axis=-1)
                       for t in taps], axis=1)
    return kind(add_block_e(values, FEATURE_NAMES), list(taps), FEATURE_NAMES + BLOCK_E)
