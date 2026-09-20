"""Run a planned injection grid and label every (injection, probe) pair (SPEC §4.2–§4.4)."""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from fidnn.inject.bitflip import state_checksum
from fidnn.inject.engine import injected_flips
from fidnn.inject.taxonomy import label, logit_shift, track
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.hooks import TapMonitor


def _tap_frame(monitor: "TapMonitor", injection_id: int, probe_index: np.ndarray) -> pd.DataFrame:
    frames = []
    for tap_id, out in monitor.outputs.items():
        df = pd.DataFrame(out.numpy(), columns=FEATURE_NAMES)
        df.insert(0, "tap_id", tap_id)
        df.insert(0, "probe_index", probe_index[:len(df)])
        df.insert(0, "injection_id", injection_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class RestoreError(RuntimeError):
    """The model did not return to its clean state — every later result would be contaminated."""


@torch.no_grad()
def _logits(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    return model(x).float().numpy()


def run(model: nn.Module, plan: pd.DataFrame, probes_x: torch.Tensor, probes_y: np.ndarray,
        clean_logits: np.ndarray, epsilon: float, num_classes: int, seed: int = 0,
        probes_per_injection: int = 32, checksum_every: int = 50,
        log: Callable[[str], None] = print,
        monitor: "TapMonitor | None" = None,
        features_out: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per (injection, probe). `model` is the fault instance, never the clean one.

    With `monitor` attached (M-3 taps), each injection's per-probe tap features are appended to
    `features_out`, keyed by injection_id — this is how fault-side features are produced (§5, §7).
    """
    rng = np.random.default_rng(seed)
    clean = state_checksum(model)
    rows, t0 = [], time.time()
    groups = list(plan.groupby("injection_id", sort=True))
    for n, (injection_id, flips) in enumerate(groups, start=1):
        idx = rng.choice(len(probes_y), size=probes_per_injection, replace=False)
        x, y, cl = probes_x[idx], probes_y[idx], clean_logits[idx]
        records = flips.to_dict("records")
        mode = records[0]["mode"]
        with injected_flips(model, records, mode) as applied:
            try:
                fl = _logits(model, x)
            except (RuntimeError, ValueError) as exc:  # a fault that breaks the forward is a CRASH
                fl = np.full_like(cl, np.nan)
                log(f"injection {injection_id}: forward raised {type(exc).__name__}: {exc}")
            if monitor is not None and features_out is not None:
                features_out.append(_tap_frame(monitor, injection_id, idx))
        labels = label(cl, fl, y, epsilon, num_classes)
        first = records[0]
        rows.append(pd.DataFrame({
            "injection_id": injection_id, "mode": mode, "attacker": first["attacker"],
            "bucket": first["bucket"], "stratum": first["stratum"], "budget": first["budget"],
            "flips_applied": len(applied), "layers": ",".join(sorted({f["layer"] for f in records})),
            "probe_index": idx, "y": y, "clean_pred": cl.argmax(1),
            "fault_pred": np.where(np.isfinite(fl).all(1), np.nan_to_num(fl).argmax(1), -1),
            "logit_shift": logit_shift(cl, fl), "label": labels, "track": track(labels),
        }))
        if n % checksum_every == 0 or n == len(groups):
            if state_checksum(model) != clean:
                raise RestoreError(f"state differs from clean after injection {injection_id}")
            log(f"{n}/{len(groups)} injections, checksum ok ({time.time() - t0:.0f}s)")
    return pd.concat(rows, ignore_index=True)


def summarise(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Taxonomy distribution and Track S share per (mode, bucket, stratum, budget) and overall."""
    by = ["mode", "bucket", "stratum", "budget"]
    counts = outcomes.pivot_table(index=by, columns="label", values="probe_index",
                                  aggfunc="count", fill_value=0)
    for lab in ("MASKED", "DEGRADED", "SDC", "CRASH"):
        if lab not in counts:
            counts[lab] = 0
    total = counts.sum(axis=1)
    counts["n"] = total
    counts["track_s_share"] = (counts.MASKED + counts.DEGRADED) / total
    return counts.reset_index()


def track_s_share(outcomes: pd.DataFrame) -> float:
    """The §12 M-2 gate quantity: MASKED + DEGRADED over all probes."""
    return float((outcomes.track == "S").mean())
