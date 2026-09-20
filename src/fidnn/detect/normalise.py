"""Robust per-feature normaliser: (x − median)/(IQR + δ), fitted on clean_fit (SPEC §5.3)."""

from dataclasses import dataclass, field

import numpy as np

from fidnn.detect.features import CleanFeatures, Features

DELTA = 1e-6


@dataclass
class RobustNormaliser:
    """Diagonal scaling only — no covariance, no whitening (C1)."""

    iqr_floor: float = 1e-8
    median: np.ndarray = field(default=None, repr=False)
    iqr: np.ndarray = field(default=None, repr=False)
    keep: np.ndarray = field(default=None, repr=False)

    def fit(self, clean_fit: CleanFeatures) -> "RobustNormaliser":
        if not isinstance(clean_fit, CleanFeatures):
            raise TypeError("the normaliser fits on clean_fit only (C2)")
        x = clean_fit.values
        self.median = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        self.iqr = q75 - q25
        self.keep = self.iqr > self.iqr_floor
        return self

    def transform(self, features: Features) -> np.ndarray:
        """(N, K, F_kept) z-scores. Dropped features are the ones with no clean spread."""
        if self.median is None:
            raise RuntimeError("normaliser is not fitted")
        z = (features.values - self.median) / (self.iqr + DELTA)
        return np.where(self.keep, z, 0.0)

    def dropped(self, features: Features) -> list[tuple[str, str]]:
        """The persisted drop list, as (tap, feature) pairs (SPEC §5.3)."""
        return [(features.taps[k], features.names[f])
                for k, f in zip(*np.where(~self.keep))]
