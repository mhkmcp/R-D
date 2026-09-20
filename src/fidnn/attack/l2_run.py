"""`fidnn attack l2`: the §10 with-vs-without-monitor table (M-8). Run on M2 INT8 first."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from fidnn.attack import adaptive
from fidnn.data.loader import Batches
from fidnn.data.prepare import load_arrays
from fidnn.detect import calibrate
from fidnn.detect.run import _clean_splits, _scored
from fidnn.models.checkpoint import load
from fidnn.provenance import sidecar
from fidnn.tables import md
from fidnn.taps.extract import saturation_thresholds
from fidnn.taps.registry import taps


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path, data_cfg: Path,
        data_dir: Path, models_dir: Path, features_dir: Path, detectors_dir: Path,
        out_dir: Path, trials: int | None = None, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    trials = trials if trials is not None else cfg["trials"]
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = arrays.of("clean_fit")
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]
    fp32 = load(model_id, seed, "fp32", models_dir)
    probes = torch.cat([xb for xb, _ in Batches(arrays.probe_x, arrays.probe_y,
                                                arrays.mean, arrays.std, 500)])
    probe_y = torch.from_numpy(arrays.probe_y)

    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    bundle = joblib.load(detectors_dir / f"{stem}.joblib")
    detector = bundle["detectors"]["D2"]
    clean = _clean_splits(features_dir, stem, bundle["tap_ids"])
    tau = calibrate.thresholds(_scored(detector, clean["clean_cal"]), (cfg["alpha"],))[cfg["alpha"]]
    log(f"{stem}: monitor threshold τ={tau:.4f} at α={cfg['alpha']}")

    rows, comparisons = [], []
    for trial in range(trials):
        rng = np.random.default_rng(1000 * seed + trial)
        attack_idx = rng.choice(len(probe_y), size=cfg["attack_batch"], replace=False)
        eval_idx = rng.choice(len(probe_y), size=cfg["eval_n"], replace=False)
        x, y = probes[attack_idx], probe_y[attack_idx]
        eval_xy = (probes[eval_idx], probe_y[eval_idx])

        free_model = load(model_id, seed, precision, models_dir, calib)
        free = adaptive.attack(free_model, fp32, x, y, cfg["max_flips"], None,
                               cfg["candidates_per_layer"], eval_xy, log=log,
                               stop_below=cfg["target_accuracy"])
        watched_model = load(model_id, seed, precision, models_dir, calib)
        sat = saturation_thresholds(watched_model, taps(model_id, tap_set), probes[:512])
        monitor = adaptive.Monitor(watched_model, taps(model_id, tap_set), detector, tau, sat)
        watched = adaptive.attack(watched_model, fp32, x, y, cfg["max_flips"], monitor,
                                  cfg["candidates_per_layer"], eval_xy, log=log,
                                  stop_below=cfg["target_accuracy"])
        rows += [free.assign(trial=trial), watched.assign(trial=trial)]
        comparisons.append(adaptive.compare(watched, free, cfg["target_accuracy"]) | {"trial": trial})

    curve = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(out_dir / f"l2_{stem}.parquet", index=False)
    record = {"model": model_id, "precision": precision, "tap_set": tap_set, "tau": tau,
              "alpha": cfg["alpha"], "comparisons": comparisons,
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"l2_{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def write_report(art: Path, out: Path) -> None:
    sides = [json.loads(p.read_text()) for p in sorted(art.glob("l2_*.json"))]
    if not sides:
        raise FileNotFoundError(f"no l2_*.json under {art}")
    comp = pd.DataFrame([c for s in sides for c in s["comparisons"]])
    side = sides[0]
    blocked = int(comp.monitor_blocked_attack.sum())
    mult = comp.cost_multiplier.dropna()
    verdict = (f"The monitor blocked the attack outright in {blocked} of {len(comp)} trials"
               + (f"; where it did not, it raised the required flip budget by "
                  f"{mult.mean():.2f}× on average." if len(mult) else "."))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# M-8 — Adaptive (L2) attacker

**Milestone:** M-8 (SPEC §12, §10) · **Model:** {side['model']} {side['precision']} ·
**Monitor:** D2 at α={side['alpha']}, τ={side['tau']:.4f} · **Commit:** `{side['git_commit']}`

The L2 attacker knows the taps and the detector, and only keeps flips that leave every monitored
score under the threshold. The unmonitored arm is the same search with the filter removed, so the
two columns differ in exactly one thing.

## Flip budget with and without the monitor

{md(comp, floatfmt=".2f")}

{verdict}

**Honest framing (§10).** A higher required budget is a real gain even when the monitor is
ultimately evadable; a blocked trial is not proof of security. Budgets are in the same units as the
M-1b table, so they compare with the published BFA figures.
""")
    print(f"[m8] wrote {out}")
