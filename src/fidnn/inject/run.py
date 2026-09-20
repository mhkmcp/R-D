"""`fidnn inject`: plan a §4.2 grid, run it, write replay records and labelled outcomes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fidnn.data.loader import Batches
from fidnn.data.prepare import load_arrays
from fidnn.inject import plan as planner
from fidnn.inject import sweep
from fidnn.inject.targets import targets
from fidnn.models.checkpoint import load
from fidnn.models.registry import MODELS
from fidnn.provenance import sidecar
from fidnn.taps.extract import saturation_thresholds
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


def _normalised(x: np.ndarray, y: np.ndarray, mean, std, batch: int = 500) -> torch.Tensor:
    return torch.cat([xb for xb, _ in Batches(x, y, mean, std, batch)])


def run(model_id: str, precision: str, mode: str, seed: int, cfg_path: Path, data_cfg: Path,
        data_dir: Path, models_dir: Path, out_dir: Path, reps: int | None = None,
        tap_set: str | None = None, features_dir: Path | None = None, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    reps = reps if reps is not None else cfg["reps"]
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = arrays.of("clean_fit")
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]

    # two instances: clean logits and features never come from a model that was injected (§4.4)
    clean_model = load(model_id, seed, precision, models_dir, calib)
    fault_model = load(model_id, seed, precision, models_dir, calib)

    probes_x = _normalised(arrays.probe_x, arrays.probe_y, arrays.mean, arrays.std)
    probes_y = arrays.probe_y
    num_classes = MODELS[model_id].num_classes
    with torch.no_grad():
        clean_logits = torch.cat([clean_model(probes_x[i:i + 500])
                                  for i in range(0, len(probes_x), 500)]).float().numpy()

    all_targets = targets(fault_model, model_id)
    grid = (planner.rnd_val_plan(all_targets, seed, reps) if mode == "rnd_val"
            else planner.plan(all_targets, mode, seed, reps, tuple(cfg["budgets"])))
    if grid.empty:
        raise ValueError(f"no targets for mode {mode} on {model_id} {precision}")
    log(f"{model_id} {precision} {mode}: {grid.injection_id.nunique()} injections, "
        f"{len(grid)} flips over {grid.layer.nunique()} layers")

    monitor, fault_features = None, ([] if tap_set else None)
    if tap_set:
        sat = saturation_thresholds(clean_model, taps(model_id, tap_set),
                                    probes_x[:cfg["sat_sample"]], cfg["sat_quantile"])
        monitor = TapMonitor(fault_model, taps(model_id, tap_set), mode="features",
                             sat_thresholds=sat)
    try:
        outcomes = sweep.run(fault_model, grid, probes_x, probes_y, clean_logits, cfg["epsilon"],
                             num_classes, seed=seed,
                             probes_per_injection=cfg["probes_per_injection"],
                             checksum_every=cfg["checksum_every"], log=log,
                             monitor=monitor, features_out=fault_features)
    finally:
        if monitor is not None:
            monitor.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_{precision}_{mode}_seed{seed}"
    grid.to_parquet(out_dir / f"{stem}_records.parquet", index=False)
    outcomes.to_parquet(out_dir / f"{stem}_outcomes.parquet", index=False)
    share = sweep.track_s_share(outcomes)
    if fault_features:
        target = features_dir or out_dir
        target.mkdir(parents=True, exist_ok=True)
        pd.concat(fault_features, ignore_index=True).to_parquet(
            target / f"{stem}_fault_features.parquet", index=False)
        log(f"{stem}: fault features written for tap set {tap_set}")
    record = {
        "model": model_id, "precision": precision, "mode": mode, "attacker": "L0",
        "tap_set": tap_set,
        "injections": int(grid.injection_id.nunique()), "probes": len(outcomes),
        "labels": outcomes.label.value_counts().to_dict(),
        "track_s_share": share,
        "bias_numel": {t.layer: t.numel for t in all_targets if t.tensor == "bias"},
        **sidecar(cfg | {"reps": reps}, seed=seed, device="cpu"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: Track S {share:.1%}, labels {record['labels']}")
    return record
