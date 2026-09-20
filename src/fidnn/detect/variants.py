"""Detector variants D1 (per tap), D2 (early fusion), D3 (late fusion) — SPEC §6.2."""

from dataclasses import dataclass, field

import numpy as np

from fidnn.detect import svdd
from fidnn.detect.features import CleanFeatures, Features
from fidnn.detect.normalise import RobustNormaliser

COMBINERS = ("max", "mean", "fpr_weighted")


def _require_clean(x: Features, what: str) -> None:
    if not isinstance(x, CleanFeatures):
        raise TypeError(f"{what} takes clean features only (C2)")


@dataclass
class D1:
    """One SVDD per tap: the per-layer baseline (H2) and the propagation map (H4)."""

    seed: int = 0
    alpha: float = 0.01
    normaliser: RobustNormaliser = field(default_factory=RobustNormaliser)
    models: dict = field(default_factory=dict, repr=False)
    selections: dict = field(default_factory=dict, repr=False)
    taps: list[str] = field(default_factory=list)

    def fit(self, clean_fit: CleanFeatures, clean_cal: CleanFeatures) -> "D1":
        _require_clean(clean_fit, "D1.fit")
        _require_clean(clean_cal, "D1.fit")
        self.normaliser.fit(clean_fit)
        zf, zc = self.normaliser.transform(clean_fit), self.normaliser.transform(clean_cal)
        self.taps = list(clean_fit.taps)
        for k, tap in enumerate(self.taps):
            sel = svdd.select(zf[:, k, :], zc[:, k, :], self.alpha, self.seed)
            self.models[tap] = svdd.fit(zf[:, k, :], sel.nu, sel.gamma)
            self.selections[tap] = sel
        return self

    def scores(self, features: Features) -> np.ndarray:
        """(N, K) — one score per tap, in registry order."""
        z = self.normaliser.transform(features)
        return np.stack([svdd.score(self.models[t], z[:, k, :])
                         for k, t in enumerate(self.taps)], axis=1)


@dataclass
class D2:
    """Early fusion: every tap's normalised features concatenated into one SVDD."""

    seed: int = 0
    alpha: float = 0.01
    normaliser: RobustNormaliser = field(default_factory=RobustNormaliser)
    model: object = field(default=None, repr=False)
    selection: svdd.Selection = field(default=None, repr=False)

    def fit(self, clean_fit: CleanFeatures, clean_cal: CleanFeatures) -> "D2":
        _require_clean(clean_fit, "D2.fit")
        _require_clean(clean_cal, "D2.fit")
        self.normaliser.fit(clean_fit)
        zf = self.normaliser.transform(clean_fit).reshape(len(clean_fit), -1)
        zc = self.normaliser.transform(clean_cal).reshape(len(clean_cal), -1)
        self.selection = svdd.select(zf, zc, self.alpha, self.seed)
        self.model = svdd.fit(zf, self.selection.nu, self.selection.gamma)
        return self

    def scores(self, features: Features) -> np.ndarray:
        z = self.normaliser.transform(features).reshape(len(features), -1)
        return svdd.score(self.model, z)


@dataclass
class D3:
    """Late fusion: per-tap scores through each tap's clean_cal CDF, then combined (§6.2)."""

    d1: D1
    combiner: str = "max"
    cdfs: dict = field(default_factory=dict, repr=False)
    tap_weights: np.ndarray = field(default=None, repr=False)
    tap_thresholds: dict = field(default_factory=dict, repr=False)

    def fit(self, clean_cal: CleanFeatures) -> "D3":
        """The CDF and the per-tap thresholds come from clean_cal only (C2, C3)."""
        _require_clean(clean_cal, "D3.fit")
        cal = self.d1.scores(clean_cal)
        for k, tap in enumerate(self.d1.taps):
            self.cdfs[tap] = np.sort(cal[:, k])
            self.tap_thresholds[tap] = float(np.quantile(cal[:, k], 1 - self.d1.alpha))
        mapped = self.map_scores(cal)
        fpr = (mapped > 1 - self.d1.alpha).mean(axis=0) + 1e-6
        self.tap_weights = (1.0 / fpr) / (1.0 / fpr).sum()
        return self

    def map_scores(self, per_tap: np.ndarray) -> np.ndarray:
        """Each tap's score → its clean_cal empirical CDF value."""
        return np.stack([np.searchsorted(self.cdfs[t], per_tap[:, k], side="right")
                         / len(self.cdfs[t]) for k, t in enumerate(self.d1.taps)], axis=1)

    def scores(self, features: Features) -> np.ndarray:
        mapped = self.map_scores(self.d1.scores(features))
        if self.combiner == "max":
            return mapped.max(axis=1)
        if self.combiner == "mean":
            return mapped.mean(axis=1)
        if self.combiner == "fpr_weighted":
            return mapped @ self.tap_weights
        raise ValueError(f"combiner must be one of {COMBINERS}")

    def first_alarm_tap(self, features: Features) -> np.ndarray:
        """H4: the first tap, in registry order, over its own threshold; -1 if none (§9.2)."""
        per_tap = self.d1.scores(features)
        over = np.stack([per_tap[:, k] > self.tap_thresholds[t]
                         for k, t in enumerate(self.d1.taps)], axis=1)
        return np.where(over.any(1), over.argmax(1), -1)
