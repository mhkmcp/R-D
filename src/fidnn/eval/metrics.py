"""Detection metrics, per track, at thresholds calibrated on clean data (SPEC §9.1)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TRACKS = ("S", "H")
CATEGORIES = ("MASKED", "DEGRADED", "SDC", "CRASH")


def tpr(fault_scores: np.ndarray, tau: float) -> float:
    return float((fault_scores > tau).mean()) if len(fault_scores) else float("nan")


def confusion(fault_scores: np.ndarray, clean_scores: np.ndarray, tau: float) -> dict:
    """Precision/recall/F1 at the calibrated threshold, with clean as the negative class."""
    tp = int((fault_scores > tau).sum())
    fn = len(fault_scores) - tp
    fp = int((clean_scores > tau).sum())
    tn = len(clean_scores) - fp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall and not np.isnan(precision) else float("nan"))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1}


def ranking(fault_scores: np.ndarray, clean_scores: np.ndarray) -> dict:
    """AUROC and AUPR — AUPR because the classes are unbalanced (§9.1)."""
    if not len(fault_scores) or not len(clean_scores):
        return {"auroc": float("nan"), "aupr": float("nan")}
    y = np.concatenate([np.ones(len(fault_scores)), np.zeros(len(clean_scores))])
    s = np.concatenate([fault_scores, clean_scores])
    return {"auroc": float(roc_auc_score(y, s)), "aupr": float(average_precision_score(y, s))}


def at_thresholds(fault_scores: np.ndarray, clean_scores: np.ndarray,
                  taus: dict[float, float]) -> list[dict]:
    rank = ranking(fault_scores, clean_scores)
    rows = []
    for alpha, tau in taus.items():
        rows.append({"alpha": alpha, "tau": tau, "n_fault": len(fault_scores),
                     "n_clean": len(clean_scores), "tpr": tpr(fault_scores, tau),
                     "measured_fpr": float((clean_scores > tau).mean()),
                     **confusion(fault_scores, clean_scores, tau), **rank})
    return rows


BREAKDOWNS = ("label", "stratum", "bucket", "budget", "mode", "precision", "attacker")


def by_track(scored: pd.DataFrame, clean_scores: np.ndarray, taus: dict[float, float],
             extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """Headline rows: one per (track, α), plus whatever `extra_keys` group by (§9.1)."""
    rows = []
    for keys, g in scored.groupby(["track", *extra_keys], dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for row in at_thresholds(g.score.to_numpy(), clean_scores, taus):
            rows.append(dict(zip(["track", *extra_keys], keys)) | row)
    return pd.DataFrame(rows)


def breakdowns(scored: pd.DataFrame, clean_scores: np.ndarray, taus: dict[float, float],
               axes: tuple[str, ...] = BREAKDOWNS) -> pd.DataFrame:
    """Every mandatory breakdown; an aggregate-only table is not acceptable (§9.1)."""
    frames = []
    for axis in axes:
        if axis not in scored:
            continue
        df = by_track(scored, clean_scores, taus, extra_keys=(axis,))
        # axes of different dtypes share one column (`budget` is an int), so it must be one type
        frames.append(df.rename(columns={axis: "value"})
                        .assign(value=lambda d: d.value.astype(str), axis=axis))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
