"""M-6 overhead study: what monitoring costs, per K and per detector variant (SPEC §9.4)."""

import json
import tracemalloc
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from fidnn.bench.throughput import time_fn
from fidnn.provenance import sidecar
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


def _forward(model: nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        model(x)


def _peak_memory(fn, device: str) -> dict:
    tracemalloc.start()
    fn()
    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    mps_mb = torch.mps.current_allocated_memory() / 2**20 if device == "mps" else None
    return {"tracemalloc_peak_kb": peak_kb, "mps_alloc_mb": mps_mb}


def measure(model: nn.Module, model_id: str, tap_set: str, device: str, precision: str,
            batches: tuple[int, ...], k_values: tuple[int, ...], timing: dict,
            detector=None, log=print) -> pd.DataFrame:
    """Absolute ms first; the percentage is derived against the stated baseline model (§9.4)."""
    tap_list = taps(model_id, tap_set)
    rows = []
    for batch in batches:
        x = torch.randn(batch, 3, 32, 32, device=device)
        forward = partial(_forward, model, x)
        base = np.median(time_fn(forward, device, timing["warmup"], timing["runs"])) * 1e3
        rows.append({"model": model_id, "device": device, "precision": precision, "batch": batch,
                     "k": 0, "stage": "model_only", "median_ms": base, "added_ms": 0.0,
                     "added_pct": 0.0, "baseline_ms": base, **_peak_memory(forward, device)})
        for k in k_values:
            subset = tap_list[:k]
            if len(subset) < k:
                continue
            for stage in ("capture", "features"):
                mon = TapMonitor(model, subset, mode=stage)
                try:
                    t = np.median(time_fn(forward, device, timing["warmup"], timing["runs"])) * 1e3
                    mem = _peak_memory(forward, device)
                finally:
                    mon.remove()
                rows.append({"model": model_id, "device": device, "precision": precision,
                             "batch": batch, "k": k, "stage": stage, "median_ms": t,
                             "added_ms": t - base, "added_pct": 100 * (t - base) / base,
                             "baseline_ms": base, **mem})
            if detector is not None:
                scoring = np.median(time_fn(partial(detector, batch), "cpu",
                                            timing["warmup"], timing["runs"])) * 1e3
                rows.append({"model": model_id, "device": device, "precision": precision,
                             "batch": batch, "k": k, "stage": "scoring", "median_ms": scoring,
                             "added_ms": scoring, "added_pct": 100 * scoring / base,
                             "baseline_ms": base})
        log(f"{model_id} {device} {precision} bs={batch}: baseline {base:.3f} ms, "
            f"{len(k_values)} tap counts measured")
    return pd.DataFrame(rows)


def throughput(df: pd.DataFrame) -> pd.DataFrame:
    """Sustained samples/s, monitor on vs off — the other half of the §9.4 table."""
    out = df[df.stage.isin(["model_only", "features"])].copy()
    out["samples_per_s"] = out.batch / (out.median_ms / 1e3)
    return out


def pareto(latency: pd.DataFrame, tpr: pd.DataFrame, latency_col: str = "added_ms",
           tpr_col: str = "tpr") -> pd.DataFrame:
    """TPR-vs-latency front over K: keep points no other point beats on both axes (§9.4, H3)."""
    joined = latency.merge(tpr, on="k", suffixes=("_lat", "_tpr"))
    keep = []
    for row in joined.itertuples():
        dominated = any(o.tpr >= getattr(row, tpr_col) and getattr(o, latency_col)
                        <= getattr(row, latency_col)
                        and (o.tpr > getattr(row, tpr_col)
                             or getattr(o, latency_col) < getattr(row, latency_col))
                        for o in joined.itertuples())
        keep.append(not dominated)
    return joined[keep].sort_values(latency_col)


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path,
        data_cfg: Path, data_dir: Path, models_dir: Path, detectors_dir: Path,
        out_dir: Path, log=print) -> dict:
    from fidnn.data.loader import Batches
    from fidnn.data.prepare import load_arrays
    from fidnn.models.checkpoint import load

    cfg = yaml.safe_load(cfg_path.read_text())
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = arrays.of("clean_fit")
    calib = [x for x, _ in Batches(fit_x[:512], fit_y[:512], arrays.mean, arrays.std, 64)]

    bundle_path = detectors_dir / f"{model_id}_{precision}_{tap_set}_seed{seed}.joblib"
    detector_kb = bundle_path.stat().st_size / 1024 if bundle_path.exists() else None
    scorer = None
    if bundle_path.exists():
        # scoring cost depends on the shape, not the values, so time the fitted SVDD directly
        d2 = joblib.load(bundle_path)["detectors"]["D2"]
        dim = d2.model.support_vectors_.shape[1]
        rng = np.random.default_rng(seed)

        def scorer(n: int) -> np.ndarray:
            return d2.model.decision_function(rng.normal(size=(n, dim)))

    frames = []
    for device in cfg["devices"]:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        if device == "mps" and precision == "int8":
            continue      # no quantised MPS kernels (§3)
        model = load(model_id, seed, precision, models_dir, calib).to(device).eval()
        frames.append(measure(model, model_id, tap_set, device, precision,
                              tuple(cfg["batch_sizes"]), tuple(cfg["k_values"]),
                              cfg["timing"], detector=scorer, log=log))
    df = pd.concat(frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    df.to_parquet(out_dir / f"{stem}_overhead.parquet", index=False)
    record = {"model": model_id, "precision": precision, "tap_set": tap_set,
              "detector_kb": detector_kb, "rows": len(df),
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"{stem}_overhead.json").write_text(json.dumps(record, indent=2, default=str))
    return record
