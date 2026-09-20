"""§9.6 ablations. Every selection here uses clean data only (C2); faults are scored, never fitted."""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from fidnn.detect import svdd
from fidnn.detect.features import BLOCK_E, CleanFeatures, FaultFeatures, Features
from fidnn.detect.normalise import RobustNormaliser
from fidnn.detect.variants import D2
from fidnn.eval.localisation import tap_buckets
from fidnn.inject.targets import BUCKETS
from fidnn.taps.features import BLOCK_A, BLOCK_B, BLOCK_C, BLOCK_D

BLOCKS = {"A": BLOCK_A, "B": BLOCK_B, "C": BLOCK_C, "D": BLOCK_D, "E": BLOCK_E}
D_SPATIAL = ["ch_mean_mean", "ch_mean_std", "ch_std_mean", "ch_std_std"]
TRAINING_BUDGETS = (500, 1000, 2000, 6000)
KERNELS = ("rbf", "poly", "linear")


def subset(features: Features, taps: list[str]) -> Features:
    idx = [features.taps.index(t) for t in taps]
    return type(features)(features.values[:, idx, :], list(taps), list(features.names))


def zeroed(features: Features, drop: list[str]) -> Features:
    """Drop a feature block by zeroing it, so widths stay comparable across ablations."""
    values = features.values.copy()
    for name in drop:
        if name in features.names:
            values[:, :, features.names.index(name)] = 0.0
    return type(features)(values, list(features.taps), list(features.names))


def _tpr_at(clean_fit: CleanFeatures, clean_cal: CleanFeatures, clean_test: CleanFeatures,
            fault: FaultFeatures, alpha: float, seed: int) -> dict:
    d2 = D2(seed=seed, alpha=alpha).fit(clean_fit, clean_cal)
    tau = float(np.quantile(d2.scores(clean_cal), 1 - alpha))
    return {"tpr": float((d2.scores(fault) > tau).mean()),
            "measured_fpr": float((d2.scores(clean_test) > tau).mean()),
            "nu": d2.selection.nu, "gamma_kind": d2.selection.gamma_kind}


def greedy_tap_selection(clean_fit: CleanFeatures, clean_cal: CleanFeatures, k_max: int,
                         alpha: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """§9.6(1) K sweep: forward selection by a **clean** criterion — lowest clean_cal FPR.

    Selecting taps by fault TPR would be a C2 violation, so no fault data enters this function.
    """
    chosen, rows = [], []
    remaining = list(clean_fit.taps)
    for k in range(1, min(k_max, len(clean_fit.taps)) + 1):
        scored = []
        for tap in remaining:
            cand = chosen + [tap]
            d2 = D2(seed=seed, alpha=alpha).fit(subset(clean_fit, cand), subset(clean_cal, cand))
            fpr = float((d2.scores(subset(clean_cal, cand))
                         > np.quantile(d2.scores(subset(clean_fit, cand)), 1 - alpha)).mean())
            scored.append((fpr, tap))
        fpr, tap = min(scored)
        chosen.append(tap)
        remaining.remove(tap)
        rows.append({"k": k, "added_tap": tap, "taps": list(chosen), "clean_cal_fpr": fpr})
    return pd.DataFrame(rows)


def tap_position(clean_fit: CleanFeatures, clean_cal: CleanFeatures, clean_test: CleanFeatures,
                 fault: FaultFeatures, alpha: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """§9.6(2): early / middle / late thirds, and logits only."""
    folded = tap_buckets(clean_fit.taps)
    groups = {b: [t for t, f in zip(clean_fit.taps, folded) if f == b] for b in BUCKETS}
    if "logits" in clean_fit.taps:
        groups["logits_only"] = ["logits"]
    rows = []
    for name, taps_ in groups.items():
        if not taps_:
            continue
        rows.append({"position": name, "taps": taps_, "k": len(taps_),
                     **_tpr_at(subset(clean_fit, taps_), subset(clean_cal, taps_),
                               subset(clean_test, taps_), subset(fault, taps_), alpha, seed)})
    return pd.DataFrame(rows)


def block_ablation(clean_fit: CleanFeatures, clean_cal: CleanFeatures, clean_test: CleanFeatures,
                   fault: FaultFeatures, alpha: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """§9.6(3): drop each block in turn, plus the D-conv → D-dense downgrade."""
    rows = [{"ablation": "full", **_tpr_at(clean_fit, clean_cal, clean_test, fault, alpha, seed)}]
    for block, names in BLOCKS.items():
        rows.append({"ablation": f"drop_{block}",
                     **_tpr_at(zeroed(clean_fit, names), zeroed(clean_cal, names),
                               zeroed(clean_test, names), zeroed(fault, names), alpha, seed)})
    rows.append({"ablation": "d_dense_downgrade",
                 **_tpr_at(zeroed(clean_fit, D_SPATIAL), zeroed(clean_cal, D_SPATIAL),
                           zeroed(clean_test, D_SPATIAL), zeroed(fault, D_SPATIAL), alpha, seed)})
    return pd.DataFrame(rows)


def training_budget(clean_fit: CleanFeatures, clean_cal: CleanFeatures,
                    clean_test: CleanFeatures, fault: FaultFeatures, alpha: float = 0.01,
                    seed: int = 0, budgets=TRAINING_BUDGETS) -> pd.DataFrame:
    """§9.6(5): how much clean data a deployer actually needs. Capped at `clean_fit` (§3.2)."""
    rows = []
    for n in budgets:
        if n > len(clean_fit):
            continue
        cut = type(clean_fit)(clean_fit.values[:n], clean_fit.taps, clean_fit.names)
        rows.append({"clean_fit_n": n,
                     **_tpr_at(cut, clean_cal, clean_test, fault, alpha, seed)})
    return pd.DataFrame(rows)


def kernel_sensitivity(clean_fit: CleanFeatures, clean_cal: CleanFeatures,
                       clean_test: CleanFeatures, fault: FaultFeatures, alpha: float = 0.01,
                       seed: int = 0, kernels=KERNELS) -> pd.DataFrame:
    """§9.6(6): RBF vs polynomial vs linear, ν fixed by the RBF selection."""
    norm = RobustNormaliser().fit(clean_fit)
    zf, zc = (norm.transform(x).reshape(len(x), -1) for x in (clean_fit, clean_cal))
    zt, za = (norm.transform(x).reshape(len(x), -1) for x in (clean_test, fault))
    sel = svdd.select(zf, zc, alpha, seed)
    rows = []
    for kernel in kernels:
        model = OneClassSVM(kernel=kernel, nu=sel.nu,
                            **({"gamma": sel.gamma} if kernel != "linear" else {})).fit(zf)
        tau = float(np.quantile(-model.decision_function(zc), 1 - alpha))
        rows.append({"kernel": kernel, "nu": sel.nu,
                     "tpr": float((-model.decision_function(za) > tau).mean()),
                     "measured_fpr": float((-model.decision_function(zt) > tau).mean())})
    return pd.DataFrame(rows)


def transfer(source_fit: CleanFeatures, source_cal: CleanFeatures, target_test: CleanFeatures,
             target_fault: FaultFeatures, alpha: float = 0.01, seed: int = 0) -> dict:
    """§9.6(7): fit on one model, apply to another. Expected to fail; that is the finding.

    Tap counts differ between architectures, so both sides are truncated to the shared width and
    the truncation is reported with the number.
    """
    k = min(len(source_fit.taps), len(target_test.taps))
    src_taps, tgt_taps = source_fit.taps[:k], target_test.taps[:k]
    d2 = D2(seed=seed, alpha=alpha).fit(subset(source_fit, src_taps), subset(source_cal, src_taps))
    tau = float(np.quantile(d2.scores(subset(source_cal, src_taps)), 1 - alpha))
    tgt_clean = subset(target_test, tgt_taps)
    tgt_fault = subset(target_fault, tgt_taps)
    return {"shared_taps": k, "source_taps": src_taps, "target_taps": tgt_taps,
            "tpr": float((d2.scores(tgt_fault) > tau).mean()),
            "measured_fpr": float((d2.scores(tgt_clean) > tau).mean())}
