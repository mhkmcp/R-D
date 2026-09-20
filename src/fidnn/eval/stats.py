"""Seeds, confidence intervals and paired comparisons (SPEC §9.5, C4)."""

import numpy as np
import pandas as pd

RESAMPLES = 2000
MIN_SEEDS = 5


def mean_ci(values, confidence: float = 0.95, resamples: int = RESAMPLES,
            seed: int = 0) -> dict:
    """Mean with a bootstrap CI. Every reported number carries n and dispersion (C4)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(resamples, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {"n": len(v), "mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_difference(a, b, confidence: float = 0.95, resamples: int = RESAMPLES,
                      seed: int = 0) -> dict:
    """Paired bootstrap over the shared evaluation set; the CI is of the difference (§9.5)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same evaluation set on both sides")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    draws = d[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    detected = not (lo <= 0 <= hi)
    return {"n": len(d), "difference": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "detected_difference": detected,
            "verdict": "difference" if detected else "no detected difference"}


def seeds_ok(seeds) -> bool:
    return len(set(seeds)) >= MIN_SEEDS


def aggregate_over_seeds(rows: pd.DataFrame, value: str, keys: list[str],
                         seed: int = 0) -> pd.DataFrame:
    """Mean ± 95 % CI across seeds, with the seed count kept in the table (C4)."""
    out = []
    for key, g in rows.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        stat = mean_ci(g[value], seed=seed)
        out.append(dict(zip(keys, key)) | {"metric": value, **stat,
                                           "seeds": g.seed.nunique() if "seed" in g else 1,
                                           "enough_seeds": seeds_ok(g.seed) if "seed" in g
                                           else False})
    return pd.DataFrame(out)
