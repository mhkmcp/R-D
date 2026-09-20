"""Thresholds and alarm rules from clean_cal only (SPEC §8, C3)."""

import numpy as np
from scipy import stats

ALPHAS = (0.05, 0.01, 0.001)
WINDOWS = ((1, 1), (8, 3), (32, 8))  # (n, m): m alarms in a window of n inferences


def thresholds(clean_cal_scores: np.ndarray, alphas=ALPHAS) -> dict[float, float]:
    """τ_α = (1−α) quantile of clean_cal. Fixed a priori; never tuned on faults (C3)."""
    return {a: float(np.quantile(clean_cal_scores, 1 - a)) for a in alphas}


def fpr_upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson upper bound — what 0.1 % means on a few thousand clean samples (§8)."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(confidence, k + 1, n - k))


def measured_fpr(scores: np.ndarray, tau: float) -> dict:
    k = int((scores > tau).sum())
    return {"n": len(scores), "alarms": k, "fpr": k / len(scores),
            "fpr_upper95": fpr_upper_bound(k, len(scores))}


def windowed_alarm(alarms: np.ndarray, n: int, m: int) -> np.ndarray:
    """m-of-n over consecutive inferences; per-inference (1,1) is always reported too (§8)."""
    if n == 1:
        return alarms.astype(bool)
    cum = np.cumsum(np.concatenate([[0], alarms.astype(int)]))
    out = np.zeros(len(alarms), dtype=bool)
    for i in range(len(alarms)):
        lo = max(0, i + 1 - n)
        out[i] = (cum[i + 1] - cum[lo]) >= m
    return out
