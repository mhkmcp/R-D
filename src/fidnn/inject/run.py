"""`fidnn inject`: plan a §4.2 grid, run it, write replay records and labelled outcomes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fidnn.cache import store
from fidnn.cache.graph import Resume, stateful_modules, traced
from fidnn.cache.suffix import SuffixRunner
from fidnn.data.loader import Batches
from fidnn.data.prepare import load_arrays, model_classes, subset_classes
from fidnn.inject import plan as planner
from fidnn.inject import sweep
from fidnn.inject.targets import targets
from fidnn.models.checkpoint import load
from fidnn.models.registry import MODELS, config_path
from fidnn.provenance import sidecar
from fidnn.taps.extract import saturation_thresholds
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


def _normalised(x: np.ndarray, y: np.ndarray, mean, std, batch: int = 500) -> torch.Tensor:
    return torch.cat([xb for xb, _ in Batches(x, y, mean, std, batch)])


EXEC_MODES = ("full", "suffix")


def default_exec(precision: str) -> str:
    """Suffix execution is bit-exact on INT8 only; FP32 GEMMs vary with batch size (cache/README)."""
    return "suffix" if precision == "int8" else "full"


def run(model_id: str, precision: str, mode: str, seed: int, cfg_path: Path, data_cfg: Path,
        data_dir: Path, models_dir: Path, out_dir: Path, reps: int | None = None,
        tap_set: str | None = None, features_dir: Path | None = None,
        exec_mode: str | None = None, cache_dir: Path = Path("artifacts/cache"),
        log=print) -> dict:
    exec_mode = exec_mode or default_exec(precision)
    if exec_mode not in EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {EXEC_MODES}")
    cfg = yaml.safe_load(cfg_path.read_text())
    reps = reps if reps is not None else cfg["reps"]
    arrays = load_arrays(data_cfg, data_dir)
    classes = model_classes(config_path(model_id))
    fit_x, fit_y = subset_classes(*arrays.of("clean_fit"), classes)
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]

    # two instances: clean logits and features never come from a model that was injected (§4.4)
    clean_model = load(model_id, seed, precision, models_dir, calib)
    fault_model = load(model_id, seed, precision, models_dir, calib)

    probe_x, probes_y = subset_classes(arrays.probe_x, arrays.probe_y, classes)
    probes_x = _normalised(probe_x, probes_y, arrays.mean, arrays.std)
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

    monitor, runner, cache, fault_features = None, None, None, ([] if tap_set else None)
    tap_list = taps(model_id, tap_set) if tap_set else []
    sat = (saturation_thresholds(clean_model, tap_list, probes_x[:cfg["sat_sample"]],
                                 cfg["sat_quantile"]) if tap_set else {})
    if exec_mode == "suffix":
        # the cache comes from the clean instance; the runner executes the fault instance (§4.4)
        clean_gm = traced(clean_model)
        starts = Resume(clean_gm).starts(stateful_modules(clean_gm))
        cache_path = cache_dir / f"{model_id}_{precision}_{tap_set or 'none'}_seed{seed}"
        cache = store.ensure(cache_path, clean_gm, probes_x, starts, tap_list, sat, log=log)
        runner = SuffixRunner(traced(fault_model), cache, tap_list, sat)
    elif tap_set:
        monitor = TapMonitor(fault_model, tap_list, mode="features", sat_thresholds=sat)
    try:
        outcomes = sweep.run(fault_model, grid, probes_x, probes_y, clean_logits, cfg["epsilon"],
                             num_classes, seed=seed,
                             probes_per_injection=cfg["probes_per_injection"],
                             checksum_every=cfg["checksum_every"], log=log,
                             monitor=monitor, features_out=fault_features, runner=runner)
    finally:
        if monitor is not None:
            monitor.remove()
    if runner is not None:
        log(f"suffix execution: {runner.stats}")

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
        "exec": {"mode": exec_mode, **({"cache": str(cache_path), "cache_key": cache.key,
                                        "stats": runner.stats} if runner else {})},
        **sidecar(cfg | {"reps": reps}, seed=seed, device="cpu"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: Track S {share:.1%}, labels {record['labels']}")
    return record
