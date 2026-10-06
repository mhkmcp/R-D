"""Classifier + tap descriptors + D2 SVDD as one exportable module (deploy/README.md)."""

import numpy as np
import torch
from torch import nn

from fidnn.detect.features import BLOCK_E
from fidnn.detect.normalise import DELTA
from fidnn.taps.features import _EPS, FEATURE_NAMES, QUANTILES


def _quantiles(sorted_flat: torch.Tensor, qs=QUANTILES) -> torch.Tensor:
    """`torch.quantile(..., interpolation="linear")` on pre-sorted rows, with static indices."""
    n = sorted_flat.shape[1]
    cols = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        cols.append(sorted_flat[:, lo] + frac * (sorted_flat[:, hi] - sorted_flat[:, lo]))
    return torch.stack(cols, 1)


def _std(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = x.mean(dim, keepdim=True)
    return ((x - mean) ** 2).mean(dim).sqrt()


def tap_features(a: torch.Tensor, sat_threshold: float) -> torch.Tensor:
    """Export-friendly twin of `fidnn.taps.features.tap_features` (conv path); parity is tested."""
    n = a.shape[0]
    if a.dim() == 2:
        a = a[:, :, None, None]
    c = a.shape[1]
    flat = a.reshape(n, -1)
    numel = flat.shape[1]
    s = torch.sort(flat, dim=1).values
    block_a = torch.cat([torch.stack([flat.mean(1), _std(flat, 1), s[:, 0], s[:, -1]], 1),
                         _quantiles(s)], 1)
    block_b = torch.stack([
        (flat == 0).float().mean(1),
        (flat > sat_threshold).float().mean(1),
        torch.isnan(flat).float().sum(1),
        torch.isinf(flat).float().sum(1),
    ], 1)
    absf = flat.abs()
    l2 = (flat * flat).sum(1).sqrt()
    linf = absf.amax(1)
    energy = (a * a).reshape(n, c, -1).sum(2)
    es = torch.sort(energy, dim=1).values
    idx = torch.arange(1, c + 1, dtype=energy.dtype)
    gini = (2 * (es * idx).sum(1)) / (c * es.sum(1) + _EPS) - (c + 1) / c
    block_c = torch.stack([absf.sum(1) / numel**0.5, l2 / numel**0.5, linf,
                           linf / (l2 + _EPS), gini], 1)
    p = energy / (energy.sum(1, keepdim=True) + _EPS)
    entropy = -torch.xlogy(p, p + _EPS).sum(1)  # p·log(p+ε) exports as NaN at p = 0 in ORT
    top = torch.flip(torch.sort(p, dim=1).values, [1]).cumsum(1)
    tops = [top[:, min(t, c) - 1] for t in (1, 2, 4)]
    per_ch = a.reshape(n, c, -1)
    ch_mean, ch_std = per_ch.mean(2), _std(per_ch, 2)
    block_d = torch.stack([entropy, *tops, ch_mean.mean(1), _std(ch_mean, 1),
                           ch_std.mean(1), _std(ch_std, 1)], 1)
    return torch.cat([block_a, block_b, block_c, block_d], 1)


class _Recorder(nn.Module):
    """Stands in for a `Tap` identity and keeps its output for the monitor."""

    def __init__(self, sink: list):
        super().__init__()
        self.sink = sink

    def forward(self, x):
        self.sink.append(x)
        return x


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent, _, name = path.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, name, module)


class MonitoredClassifier(nn.Module):
    """uint8 NHWC images → logits, probabilities, class, D2 score, alarm per α.

    Every constant comes from the fitted artifacts: train-split channel stats, clean_fit saturation
    levels and normaliser, the D2 SVDD and its clean_cal thresholds. Nothing is fitted here.
    """

    def __init__(self, model: nn.Module, tap_paths: list[str], sat: list[float], mean, std,
                 normaliser, svm, taus: list[float]):
        super().__init__()
        self._taps: list[torch.Tensor] = []
        self.model = model.eval()
        for path in tap_paths:
            _set_module(self.model, path, _Recorder(self._taps))
        self.sat = list(sat)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        f64 = torch.float64
        self.register_buffer("median", torch.tensor(normaliser.median, dtype=f64))
        self.register_buffer("iqr", torch.tensor(normaliser.iqr, dtype=f64))
        self.register_buffer("keep", torch.tensor(normaliser.keep))
        self.register_buffer("sv", torch.tensor(svm.support_vectors_, dtype=f64))
        self.register_buffer("coef", torch.tensor(svm.dual_coef_[0], dtype=f64))
        self.register_buffer("intercept", torch.tensor(float(svm.intercept_[0]), dtype=f64))
        self.register_buffer("gamma", torch.tensor(float(svm._gamma), dtype=f64))
        self.register_buffer("taus", torch.tensor(taus, dtype=f64))

    def features(self) -> torch.Tensor:
        """(N, K, 29): Blocks A–D per tap, then Block E against the previous tap (SPEC §5.2)."""
        v = torch.stack([tap_features(a, s) for a, s in zip(self._taps, self.sat)], 1)
        names = ["l2_rms", "energy_entropy", "sat_frac"]
        l2, ent, sat = (v[..., FEATURE_NAMES.index(k)] for k in names)
        prev = [torch.cat([x[:, :1], x[:, :-1]], 1) for x in (l2, ent, sat)]
        e = torch.stack([l2 / (prev[0] + _EPS), ent / (prev[1] + _EPS), sat - prev[2]], -1)
        assert e.shape[-1] == len(BLOCK_E)
        return torch.cat([v, e], -1)

    def forward(self, images: torch.Tensor):
        self._taps.clear()
        x = images.permute(0, 3, 1, 2).float() / 255
        logits = self.model((x - self.mean) / self.std)
        f = self.features().double()
        self._taps.clear()
        z = torch.where(self.keep, (f - self.median) / (self.iqr + DELTA),
                        torch.zeros_like(f)).reshape(f.shape[0], -1)
        d2 = (z * z).sum(1, keepdim=True) - 2 * z @ self.sv.T + (self.sv * self.sv).sum(1)
        score = -((torch.exp(-self.gamma * d2) * self.coef).sum(1) + self.intercept)
        score = torch.where(torch.isfinite(score), score, torch.full_like(score, float("inf")))
        alarms = score[:, None] > self.taus[None, :]
        return logits, torch.softmax(logits, 1), logits.argmax(1), score, alarms
