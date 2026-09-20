"""Outcome taxonomy: MASKED / DEGRADED / SDC / CRASH and their tracks (SPEC §4.3)."""

import numpy as np

LABELS = ("MASKED", "DEGRADED", "SDC", "CRASH")
TRACK = {"MASKED": "S", "DEGRADED": "S", "SDC": "H", "CRASH": "H"}


def logit_shift(clean: np.ndarray, fault: np.ndarray) -> np.ndarray:
    """L-inf shift per probe; non-finite faulty logits give inf (CRASH is decided first)."""
    with np.errstate(invalid="ignore"):
        return np.abs(np.nan_to_num(fault, nan=np.inf) - clean).max(axis=1)


def label(clean: np.ndarray, fault: np.ndarray, y: np.ndarray, epsilon: float,
          num_classes: int) -> np.ndarray:
    """One label per probe. CRASH first, so a NaN never compares as 'prediction unchanged'.

    `CRASH` covers both of §4.3's numerical-failure cases: non-finite outputs per probe, and an
    injection whose accuracy over its own probes collapses to chance (an injection-level property,
    so it labels every probe of that injection).
    """
    finite = np.isfinite(fault).all(axis=1)
    pred_c, pred_f = clean.argmax(1), np.where(finite, np.nan_to_num(fault).argmax(1), -1)
    acc_f = float((pred_f[finite] == y[finite]).mean()) if finite.any() else 0.0
    collapsed = acc_f <= 1.0 / num_classes and (pred_c == y).mean() > 1.0 / num_classes

    out = np.full(len(clean), "MASKED", dtype=object)
    out[logit_shift(clean, fault) >= epsilon] = "DEGRADED"
    out[pred_f != pred_c] = "SDC"
    out[~finite] = "CRASH"
    if collapsed:
        out[:] = "CRASH"
    return out


def track(labels: np.ndarray) -> np.ndarray:
    return np.array([TRACK[x] for x in labels], dtype=object)
