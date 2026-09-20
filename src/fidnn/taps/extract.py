"""`fidnn extract`: clean per-tap features for the detector splits (SPEC §5.2, §5.3, §7)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from fidnn.data.loader import Batches
from fidnn.data.prepare import load_arrays
from fidnn.models.checkpoint import load
from fidnn.provenance import sidecar
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps

CLEAN_SPLITS = ("clean_fit", "clean_cal", "clean_test")


@torch.no_grad()
def saturation_thresholds(model: nn.Module, tap_list, x: torch.Tensor,
                          quantile: float = 0.999) -> dict[str, float]:
    """Per-tap clean quantile used as Block B's saturation level (SPEC §5.2).

    Fitted on a `clean_fit` subsample: an exact quantile over every clean_fit activation would
    need ~10^8 values per tap. C2: clean data only.
    """
    with TapMonitor(model, tap_list, mode="capture") as mon:
        model(x)
        return {tap_id: float(torch.quantile(out.flatten().float(), quantile))
                for tap_id, out in mon.outputs.items()}


@torch.no_grad()
def features(model: nn.Module, tap_list, x: torch.Tensor, sat: dict[str, float],
             batch: int = 250) -> pd.DataFrame:
    """One row per (sample, tap): the 26-wide Block A–D descriptor (Block E comes at fusion)."""
    rows = []
    with TapMonitor(model, tap_list, mode="features", sat_thresholds=sat) as mon:
        for start in range(0, len(x), batch):
            model(x[start:start + batch])
            for tap_id, out in mon.outputs.items():
                df = pd.DataFrame(out.numpy(), columns=FEATURE_NAMES)
                df.insert(0, "tap_id", tap_id)
                df.insert(0, "index", np.arange(start, start + len(df)))
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path, data_cfg: Path,
        data_dir: Path, models_dir: Path, out_dir: Path, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = arrays.of("clean_fit")
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]
    model = load(model_id, seed, precision, models_dir, calib)
    tap_list = taps(model_id, tap_set)

    def tensor(x, y):
        return torch.cat([xb for xb, _ in Batches(x, y, arrays.mean, arrays.std, 500)])

    sat = saturation_thresholds(model, tap_list, tensor(*[a[:cfg["sat_sample"]]
                                                          for a in (fit_x, fit_y)]),
                                cfg["sat_quantile"])
    log(f"{model_id} {precision} {tap_set}: {len(tap_list)} taps, saturation thresholds fitted")

    frames = []
    for split in CLEAN_SPLITS:
        x, y = arrays.of(split)
        df = features(model, tap_list, tensor(x, y), sat, cfg["batch"])
        df.insert(0, "split", split)
        frames.append(df)
        log(f"  {split}: {len(x)} samples × {len(tap_list)} taps")
    out = pd.concat(frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    out.to_parquet(out_dir / f"{stem}_clean.parquet", index=False)
    record = {"model": model_id, "precision": precision, "tap_set": tap_set,
              "taps": [t.tap_id for t in tap_list], "feature_names": FEATURE_NAMES,
              "saturation": {"quantile": cfg["sat_quantile"], "fitted_on": "clean_fit",
                             "sample": cfg["sat_sample"], "thresholds": sat},
              "counts": {s: int((out.split == s).sum()) for s in CLEAN_SPLITS},
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: {len(out):,} rows")
    return record
