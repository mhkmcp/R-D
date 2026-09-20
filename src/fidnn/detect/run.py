"""`fidnn fit` / `fidnn calibrate`: fit detectors on clean splits, calibrate, sanity-check (M-4)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from fidnn.detect import calibrate
from fidnn.detect.baselines import FeatureSpaceBaselines
from fidnn.detect.features import CleanFeatures, FaultFeatures, from_frame
from fidnn.detect.variants import D1, D2, D3
from fidnn.provenance import sidecar
from fidnn.tables import md
from fidnn.taps.registry import taps

SANITY_BAND = (0.3, 3.0)  # SPEC §12 M-4: calibrated FPR within 0.3×–3× nominal


def _clean_splits(features_dir: Path, stem: str, tap_ids: list[str]) -> dict[str, CleanFeatures]:
    df = pd.read_parquet(features_dir / f"{stem}_clean.parquet")
    return {split: from_frame(g, tap_ids, CleanFeatures)
            for split, g in df.groupby("split")}


def _fault_features(features_dir: Path, faults_dir: Path, model_id: str, precision: str,
                    seed: int, tap_ids: list[str]) -> tuple[FaultFeatures, pd.DataFrame] | None:
    """Fault features and their labels, joined on (injection_id, probe_index). Scoring only."""
    paths = sorted(features_dir.glob(f"{model_id}_{precision}_*_seed{seed}_fault_features.parquet"))
    if not paths:
        return None
    feats, outs = [], []
    for p in paths:
        mode = p.name.split("_")[2]
        out = pd.read_parquet(faults_dir / f"{model_id}_{precision}_{mode}_seed{seed}"
                                           "_outcomes.parquet")
        feats.append(pd.read_parquet(p).assign(mode=mode))
        outs.append(out.assign(mode=mode))
    f = pd.concat(feats, ignore_index=True)
    labels = (pd.concat(outs, ignore_index=True)
              .set_index(["mode", "injection_id", "probe_index"])[["label", "track"]])
    keys = (f[["mode", "injection_id", "probe_index"]].drop_duplicates()
            .set_index(["mode", "injection_id", "probe_index"]))
    return (from_frame(f, tap_ids, FaultFeatures,
                       index_cols=("mode", "injection_id", "probe_index")),
            labels.loc[keys.index].reset_index())


def fit_all(clean: dict[str, CleanFeatures], seed: int, alpha: float) -> dict:
    d1 = D1(seed=seed, alpha=alpha).fit(clean["clean_fit"], clean["clean_cal"])
    return {
        "D1": d1,
        "D2": D2(seed=seed, alpha=alpha).fit(clean["clean_fit"], clean["clean_cal"]),
        "D3_max": D3(d1, "max").fit(clean["clean_cal"]),
        "D3_mean": D3(d1, "mean").fit(clean["clean_cal"]),
        "D3_fpr_weighted": D3(d1, "fpr_weighted").fit(clean["clean_cal"]),
        "baselines": FeatureSpaceBaselines(seed=seed).fit(clean["clean_fit"]),
    }


def _scored(detector, features) -> np.ndarray:
    scores = detector.scores(features)
    return scores.max(axis=1) if scores.ndim == 2 else scores


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path,
        features_dir: Path, faults_dir: Path, out_dir: Path, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    tap_ids = [t.tap_id for t in taps(model_id, tap_set)]
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    clean = _clean_splits(features_dir, stem, tap_ids)
    log(f"{stem}: " + ", ".join(f"{k} {len(v)}" for k, v in clean.items()))

    detectors = fit_all(clean, seed, cfg["alpha"])
    faults = _fault_features(features_dir, faults_dir, model_id, precision, seed, tap_ids)

    rows = []
    for name, det in detectors.items():
        if name == "baselines":
            continue
        cal_scores = _scored(det, clean["clean_cal"])
        taus = calibrate.thresholds(cal_scores, cfg["alphas"])
        test_scores = _scored(det, clean["clean_test"])
        for alpha, tau in taus.items():
            row = {"detector": name, "alpha": alpha, "tau": tau,
                   **calibrate.measured_fpr(test_scores, tau)}
            row["fpr_ratio"] = row["fpr"] / alpha if alpha else np.nan
            if faults is not None:
                fault_scores = _scored(det, faults[0])
                for track, g in faults[1].groupby("track"):
                    row[f"tpr_{track}"] = float((fault_scores[g.index] > tau).mean())
            rows.append(row)
    table = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{stem}.joblib"
    joblib.dump({"detectors": detectors, "tap_ids": tap_ids, "alpha": cfg["alpha"]}, bundle)
    table.to_parquet(out_dir / f"{stem}_calibration.parquet", index=False)
    record = {
        "model": model_id, "precision": precision, "tap_set": tap_set, "taps": tap_ids,
        "thesis_result": False if model_id == "m1" else None,
        "selection": {"D2": vars(detectors["D2"].selection) | {"trials": "see parquet"},
                      "D1": {t: s.nu for t, s in detectors["D1"].selections.items()}},
        "dropped_features": detectors["D2"].normaliser.dropped(clean["clean_fit"]),
        "detector_kb": round(bundle.stat().st_size / 1024, 1),
        "calibration": table.to_dict("records"),
        "fault_features": None if faults is None else len(faults[0]),
        **sidecar(cfg, seed=seed, device="cpu"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: detectors {record['detector_kb']} KB, "
        f"FPR ratios {table.fpr_ratio.round(2).tolist()}")
    return record


def write_report(art: Path, out: Path, model_id: str = "m1") -> None:
    """docs/M4_sanity.md — a pipeline sanity gate that produces no thesis number (SPEC §12)."""
    sides = [json.loads(p.read_text()) for p in sorted(art.glob(f"{model_id}_*.json"))]
    if not sides:
        raise FileNotFoundError(f"no {model_id}_*.json under {art}")
    table = pd.concat([pd.DataFrame(s["calibration"]).assign(precision=s["precision"],
                                                             tap_set=s["tap_set"])
                       for s in sides], ignore_index=True)
    in_band = table.fpr_ratio.between(*SANITY_BAND)
    tpr_cols = [c for c in table.columns if c.startswith("tpr_")]
    verdict = (f"**M-4 sanity PASSED.** Every calibrated FPR on `clean_test` is within "
               f"{SANITY_BAND[0]}×–{SANITY_BAND[1]}× of nominal."
               if in_band.all() else
               f"**M-4 sanity NOT MET.** {int((~in_band).sum())} of {len(table)} calibrated FPRs "
               f"fall outside {SANITY_BAND[0]}×–{SANITY_BAND[1]}× of nominal — the pipeline is "
               f"miscalibrated; fix before M-5.")
    if not tpr_cols:
        verdict += ("\n\nNo fault features were present, so only the calibration half of the gate "
                    "ran. Generate them with `fidnn inject --tap-set …`.")

    side = sides[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# M-4 — Detector pipeline sanity check ({model_id.upper()})

**Milestone:** M-4 (SPEC §12) · **Commit:** `{side['git_commit']}` · **Generated:** {side['timestamp']}

> **This page produces no thesis number.** M-4 is a pipeline sanity gate on {model_id.upper()}, the
> smoke-test model (SPEC §3). The first reported detector numbers come from M-5 on M2.

## Verdict

{verdict}

## Calibration and detection

{md(table, floatfmt=".4f")}

`alpha` is the nominal FPR fixed a priori (§8, C3); `fpr` is measured on `clean_test`;
`fpr_upper95` is the Clopper-Pearson bound, which is what 0.1 % means on this many clean samples.
{"TPR columns are per track and never merged (§4.3)." if tpr_cols else ""}

## Fitted detectors

{md(pd.DataFrame([{k: s[k] for k in ('model', 'precision', 'tap_set', 'detector_kb',
                                     'fault_features')} for s in sides]))}

Dropped features (IQR below the floor on `clean_fit`, §5.3): {len(side['dropped_features'])}.
""")
    print(f"[m4] wrote {out}")
