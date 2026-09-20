"""H4: does the first alarm point at the injected layer? (SPEC §9.2)"""

import numpy as np
import pandas as pd
from scipy import stats

from fidnn.inject.targets import BUCKETS


def spearman(injected_index: np.ndarray, first_alarm_index: np.ndarray) -> dict:
    """ρ between injected layer index and first-alarm tap index; H4 fails if it is not > 0."""
    mask = (first_alarm_index >= 0) & ~np.isnan(injected_index)
    if mask.sum() < 3 or len(np.unique(first_alarm_index[mask])) < 2:
        return {"n": int(mask.sum()), "rho": float("nan"), "p_value": float("nan"),
                "greater_than_zero": False}
    rho, p = stats.spearmanr(injected_index[mask], first_alarm_index[mask], alternative="greater")
    return {"n": int(mask.sum()), "rho": float(rho), "p_value": float(p),
            "greater_than_zero": bool(p < 0.05 and rho > 0)}


def tap_buckets(tap_ids: list[str]) -> list[str]:
    """Registry order folded into early/middle/late thirds, so alarms and injections compare."""
    n = len(tap_ids)
    return [BUCKETS[min(i * len(BUCKETS) // n, len(BUCKETS) - 1)] for i in range(n)]


def alarm_buckets(first_alarm_index: np.ndarray, tap_ids: list[str]) -> pd.Series:
    folded = tap_buckets(tap_ids)
    return pd.Series([folded[i] if i >= 0 else "none" for i in first_alarm_index])


def bucket_confusion(injected_bucket: pd.Series, alarm_bucket: pd.Series) -> pd.DataFrame:
    """Injected bucket × first-alarm bucket, counted per probe (§9.2)."""
    return (pd.crosstab(injected_bucket, alarm_bucket)
            .reindex(index=list(BUCKETS), columns=[*BUCKETS, "none"], fill_value=0))


def propagation_depth(injected_index: np.ndarray, first_alarm_index: np.ndarray) -> dict:
    """How many taps downstream the alarm appears; negative means it fired upstream."""
    mask = first_alarm_index >= 0
    depth = first_alarm_index[mask] - injected_index[mask]
    if not len(depth):
        return {"n": 0, "mean_depth": float("nan"), "median_depth": float("nan")}
    return {"n": int(mask.sum()), "mean_depth": float(depth.mean()),
            "median_depth": float(np.median(depth))}


def pre_add_share(first_alarm_taps: pd.Series, pre_add_taps: set[str]) -> float:
    """ResNet only: do first alarms concentrate on the pre-addition tap? (§5.1, §9.2)"""
    fired = first_alarm_taps[first_alarm_taps != "none"]
    return float(fired.isin(pre_add_taps).mean()) if len(fired) else float("nan")
