"""Forward-hook monitor over registered taps (SPEC §5.1)."""

from collections.abc import Sequence

import torch
import torch.nn as nn

from fidnn.taps.features import tap_features
from fidnn.taps.registry import TapInfo

MODES = ("capture", "features")


class TapMonitor:
    """Forward hooks on `taps`; see `nn.Module.register_forward_hook`. Output in `self.outputs`."""

    def __init__(self, model: nn.Module, taps: Sequence[TapInfo], mode: str = "features",
                 sat_thresholds: dict[str, float] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.sat = sat_thresholds or {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = {t.tap_id: 0 for t in taps}
        modules = dict(model.named_modules())
        missing = [t.path for t in taps if t.path not in modules]
        if missing:
            raise KeyError(f"tap modules not found in model: {missing}")
        self._handles = [
            modules[t.path].register_forward_hook(self._hook(t.tap_id), always_call=True)
            for t in taps
        ]

    def _hook(self, tap_id: str):
        def hook(_module, _inputs, output):
            self.calls[tap_id] += 1
            out = output.dequantize() if output.is_quantized else output
            if self.mode == "capture":
                self.outputs[tap_id] = out.detach()
            else:
                self.outputs[tap_id] = tap_features(
                    out.detach(), self.sat.get(tap_id, float("inf")))
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
