"""`fidnn eval`: score every detector against the fault population and build the results tables."""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from fidnn.detect import calibrate
from fidnn.detect.run import _clean_splits, _fault_features, _scored
from fidnn.eval import leakage, localisation, metrics, splits, stats
from fidnn.inject.plan import STRATA
from fidnn.inject.targets import BUCKETS
from fidnn.provenance import sidecar
from fidnn.taps.registry import taps

OUTPUT_ONLY_COLS = ("out_msp", "out_entropy", "out_margin", "out_energy")


def _grid_axes(fault: pd.DataFrame, width: int) -> dict[str, set]:
    return {"bucket": set(BUCKETS) & set(fault.bucket.unique()) or set(BUCKETS),
            "stratum": set(STRATA[width]) & set(fault.stratum.unique()),
            "budget": set(fault.budget.unique())}


def _clean_output_only(features_dir: Path, stem: str) -> pd.DataFrame | None:
    p = features_dir / f"{stem}_clean_outputs.parquet"
    return pd.read_parquet(p) if p.exists() else None


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path,
        features_dir: Path, faults_dir: Path, detectors_dir: Path, data_dir: Path,
        out_dir: Path, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    tap_ids = [t.tap_id for t in taps(model_id, tap_set)]
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    bundle = joblib.load(detectors_dir / f"{stem}.joblib")
    detectors = bundle["detectors"]
    clean = _clean_splits(features_dir, stem, tap_ids)
    faults = _fault_features(features_dir, faults_dir, model_id, precision, seed, tap_ids)
    if faults is None:
        raise FileNotFoundError("no fault features: run `fidnn inject --tap-set …` first")
    fault_features, fault_labels = faults

    fault_labels = fault_labels.assign(fault_split=splits.assign(fault_labels, seed=seed),
                                       is_gen=splits.gen_mask(fault_labels))
    data_meta = json.loads((data_dir / "cifar10.json").read_text())
    width = 8 if precision == "int8" else 32
    leakage.assert_clean(fault_labels, clean_splits={}, grid_axes=_grid_axes(fault_labels, width),
                         data_integrity=data_meta["integrity"])
    log(f"{stem}: leakage guard passed, {len(fault_labels):,} fault probes")

    clean_outputs = _clean_output_only(features_dir, stem)
    rows, headline = [], []
    for name, det in detectors.items():
        if name == "baselines":
            continue
        cal = _scored(det, clean["clean_cal"])
        taus = calibrate.thresholds(cal, cfg["alphas"])
        clean_scores = _scored(det, clean["clean_test"])
        scored = fault_labels.assign(score=_scored(det, fault_features), detector=name,
                                     precision=precision, model=model_id, tap_set=tap_set,
                                     seed=seed)
        held = scored[scored.fault_split == "fault_test"]
        headline.append(metrics.by_track(held, clean_scores, taus).assign(detector=name))
        rows.append(metrics.breakdowns(held, clean_scores, taus).assign(detector=name))
        gen = held[held.is_gen]
        if len(gen):   # unseen configurations: its own row, never merged into the headline (§7)
            headline.append(metrics.by_track(gen, clean_scores, taus)
                            .assign(detector=name, subset="fault_gen"))

    # §6.4 output-only baselines, on the same probes and the same clean split
    for col in OUTPUT_ONLY_COLS:
        if col not in fault_labels or clean_outputs is None:
            continue
        cal = clean_outputs.loc[clean_outputs.split == "clean_cal", col[4:]].to_numpy()
        clean_scores = clean_outputs.loc[clean_outputs.split == "clean_test", col[4:]].to_numpy()
        taus = calibrate.thresholds(cal, cfg["alphas"])
        scored = fault_labels.assign(score=fault_labels[col])
        held = scored[scored.fault_split == "fault_test"]
        headline.append(metrics.by_track(held, clean_scores, taus).assign(detector=col[4:],
                                                                          family="output_only"))

    headline_df = pd.concat(headline, ignore_index=True).assign(
        model=model_id, precision=precision, tap_set=tap_set, seed=seed)
    breakdown_df = pd.concat(rows, ignore_index=True).assign(
        model=model_id, precision=precision, tap_set=tap_set, seed=seed)

    loc = {}
    if "D3_max" in detectors:
        # Spearman is rank-based, so the injected bucket index compares directly with the tap index
        first = detectors["D3_max"].first_alarm_tap(fault_features)
        injected = fault_labels.bucket.map({b: i for i, b in enumerate(BUCKETS)}).to_numpy()
        alarm_bucket = localisation.alarm_buckets(first, tap_ids)
        alarm_bucket.index = fault_labels.index
        loc = {"spearman": localisation.spearman(injected, first),
               "propagation": localisation.propagation_depth(injected, first),
               "confusion": localisation.bucket_confusion(fault_labels.bucket,
                                                          alarm_bucket).to_dict(),
               "pre_add_share": localisation.pre_add_share(
                   pd.Series([tap_ids[i] if i >= 0 else "none" for i in first]),
                   {t.tap_id for t in taps(model_id, tap_set) if t.pre_add})}

    out_dir.mkdir(parents=True, exist_ok=True)
    headline_df.to_parquet(out_dir / f"{stem}_headline.parquet", index=False)
    breakdown_df.to_parquet(out_dir / f"{stem}_breakdowns.parquet", index=False)
    record = {"model": model_id, "precision": precision, "tap_set": tap_set, "seed": seed,
              "fault_probes": len(fault_labels),
              "fault_split_counts": fault_labels.fault_split.value_counts().to_dict(),
              "localisation": loc, "leakage": "passed",
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"{stem}_eval.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: headline rows {len(headline_df)}, breakdown rows {len(breakdown_df)}")
    return record


def aggregate(results_dir: Path, seed_keys=("model", "precision", "tap_set", "detector", "track",
                                            "alpha")) -> pd.DataFrame:
    """Mean ± 95 % CI over seeds for the headline metric (C4: n and dispersion always)."""
    frames = [pd.read_parquet(p) for p in sorted(results_dir.glob("*_headline.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no *_headline.parquet under {results_dir}")
    allrows = pd.concat(frames, ignore_index=True)
    return stats.aggregate_over_seeds(allrows, "tpr", list(seed_keys))
