"""L2 detector-aware attacker: BFA's search, constrained to stay under the alarm (SPEC §10)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from fidnn.attack import bfa
from fidnn.detect.features import FaultFeatures, add_block_e
from fidnn.inject.bitflip import flip_quantized_weight_
from fidnn.inject.targets import is_quantized
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.hooks import TapMonitor


@dataclass
class Monitor:
    """The detector as the attacker sees it: taps, a score function and a threshold (§10)."""

    model: nn.Module
    tap_list: list
    detector: object
    tau: float
    sat: dict | None = None

    def score(self, x: torch.Tensor) -> np.ndarray:
        with TapMonitor(self.model, self.tap_list, mode="features",
                        sat_thresholds=self.sat) as mon, torch.no_grad():
            self.model(x)
            values = np.stack([mon.outputs[t.tap_id].numpy() for t in self.tap_list], axis=1)
        feats = FaultFeatures(add_block_e(values, FEATURE_NAMES),
                              [t.tap_id for t in self.tap_list], FEATURE_NAMES + [
                                  "e_l2_ratio", "e_entropy_ratio", "e_sat_diff"])
        s = self.detector.scores(feats)
        return s.max(axis=1) if s.ndim == 2 else s

    def evades(self, x: torch.Tensor) -> bool:
        """True when every monitored inference stays under the calibrated threshold."""
        return bool((self.score(x) <= self.tau).all())


def attack(int8: nn.Module, fp32: nn.Module, x: torch.Tensor, y: torch.Tensor, budget: int,
           monitor: Monitor | None, candidates_per_layer: int = 10,
           eval_xy: tuple[torch.Tensor, torch.Tensor] | None = None, log=print,
           stop_below: float | None = None) -> pd.DataFrame:
    """Greedy loss-gain search, filtered by the monitor. `monitor=None` is the unmonitored arm.

    Reports the same columns as L1, so the with/without comparison is like for like (§10).
    """
    layers = {n: m for n, m in int8.named_modules() if is_quantized(m)}
    rows = []
    for step in range(1, budget + 1):
        twin, convs = bfa.surrogate(fp32, int8)
        twin.zero_grad(set_to_none=True)
        F.cross_entropy(twin(x), y).backward()
        with torch.no_grad():
            before = float(F.cross_entropy(int8(x), y))

        best, best_loss, rejected = None, before, 0
        for name, module in layers.items():
            grad = convs[name].weight.grad
            if grad is None:
                continue
            for idx, bit, _ in bfa.rank_candidates(module, grad.detach().flatten(),
                                                   candidates_per_layer):
                flip_quantized_weight_(module, [idx], [bit])
                with torch.no_grad():
                    loss = float(F.cross_entropy(int8(x), y))
                accepted = monitor is None or monitor.evades(x)
                flip_quantized_weight_(module, [idx], [bit])
                if not accepted:
                    rejected += 1
                    continue
                if loss > best_loss:
                    best, best_loss = bfa.Flip(name, idx, bit), loss
        if best is None:
            log(f"step {step}: no flip both increases loss and stays under the alarm; stopping")
            break

        flip_quantized_weight_(layers[best.layer], [best.flat_index], [best.bit])
        row = {"step": step, "attacker": "L2" if monitor else "L1_unmonitored",
               "layer": best.layer, "flat_index": best.flat_index, "bit": best.bit,
               "loss_before": before, "loss_after": best_loss, "rejected_candidates": rejected}
        if eval_xy is not None:
            row["accuracy"] = bfa.accuracy(int8, *eval_xy)
        rows.append(row)
        log(f"step {step}: {best.layer}[{best.flat_index}] bit {best.bit}, "
            f"loss {before:.3f} → {best_loss:.3f}, {rejected} candidates refused by the monitor")
        if stop_below is not None and row.get("accuracy", float("inf")) < stop_below:
            break
    return pd.DataFrame(rows)


def compare(with_monitor: pd.DataFrame, without: pd.DataFrame, target: float) -> dict:
    """Success rate and required flip budget, with vs without the monitor (§10)."""
    def budget_to_target(df):
        if "accuracy" not in df or not (df.accuracy < target).any():
            return None
        return int(df.loc[df.accuracy < target, "step"].min())

    constrained, free = budget_to_target(with_monitor), budget_to_target(without)
    return {
        "target_accuracy": target,
        "flips_unmonitored": free,
        "flips_monitored": constrained,
        "cost_multiplier": (constrained / free) if constrained and free else None,
        "monitor_blocked_attack": constrained is None and free is not None,
        "rejected_candidates": int(with_monitor.get("rejected_candidates", pd.Series()).sum()),
    }
