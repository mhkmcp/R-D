"""RBF SVDD (ν-one-class SVM) and its clean-only hyperparameter selection (SPEC §6.1, §6.3)."""

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM

NUS = (0.001, 0.005, 0.01, 0.05, 0.1)
GAMMA_SCALES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SV_BAND = (0.5, 3.0)  # support-vector fraction must stay within this multiple of ν


def gamma_median(x: np.ndarray, seed: int = 0, subsample: int = 1000) -> float:
    """1 / median‖xᵢ − xⱼ‖² on a seeded clean subsample (SPEC §6.3)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(subsample, len(x)), replace=False)
    s = x[idx]
    d2 = ((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices(len(s), k=1)])
    return float(1.0 / med) if med > 0 else 1.0


def fit(x: np.ndarray, nu: float, gamma) -> OneClassSVM:
    return OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(x)


def score(model: OneClassSVM, x: np.ndarray) -> np.ndarray:
    """SPEC §6.1: s(x) = −decision_function(x); > 0 means outside the boundary."""
    return -model.decision_function(x)


@dataclass(frozen=True)
class Selection:
    nu: float
    gamma: float
    gamma_kind: str
    clean_cal_fpr: float
    sv_fraction: float
    trials: list[dict]


def select(clean_fit: np.ndarray, clean_cal: np.ndarray, alpha: float = 0.01,
           seed: int = 0, nus=NUS, gamma_scales=GAMMA_SCALES) -> Selection:
    """Minimum clean-cal FPR subject to an SV-fraction band — clean data only (C2, C3).

    Selecting by fault scores or test AUROC would be a C2 violation; nothing here sees a fault.
    """
    gm = gamma_median(clean_fit, seed=seed)
    grid = [("scale", "scale")] + [(f"{k}x_median", k * gm) for k in gamma_scales]
    trials, best = [], None
    for nu in nus:
        for kind, gamma in grid:
            model = fit(clean_fit, nu, gamma)
            sv = model.support_vectors_.shape[0] / len(clean_fit)
            tau = float(np.quantile(score(model, clean_fit), 1 - alpha))
            fpr = float((score(model, clean_cal) > tau).mean())
            in_band = SV_BAND[0] * nu <= sv <= SV_BAND[1] * nu
            trials.append({"nu": nu, "gamma_kind": kind, "gamma": gamma, "sv_fraction": sv,
                           "clean_cal_fpr": fpr, "in_band": in_band})
            if in_band and (best is None or (fpr, abs(sv - nu)) < (best[0], best[1])):
                best = (fpr, abs(sv - nu), nu, kind, gamma, sv)
    if best is None:  # no pair held the band: fall back to the closest SV fraction, and say so
        t = min(trials, key=lambda t: abs(t["sv_fraction"] - t["nu"]))
        return Selection(t["nu"], t["gamma"], t["gamma_kind"], t["clean_cal_fpr"],
                         t["sv_fraction"], trials)
    fpr, _, nu, kind, gamma, sv = best
    return Selection(nu, gamma, kind, fpr, sv, trials)
