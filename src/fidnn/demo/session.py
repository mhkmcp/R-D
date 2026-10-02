"""Live fault demo: flip bits, run the fault instance, score it with the fitted detectors."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from fidnn.cache import store
from fidnn.cache.graph import Resume, stateful_modules, traced
from fidnn.cache.suffix import SuffixRunner
from fidnn.detect.baselines import output_only_scores
from fidnn.detect.features import BLOCK_E, FaultFeatures, Features, add_block_e
from fidnn.inject.bitflip import flip_bits_, state_checksum
from fidnn.inject.engine import injected_flips
from fidnn.inject.plan import STRATA, sample_flips, width_of
from fidnn.inject.targets import BUCKETS, Target, for_mode, targets
from fidnn.inject.taxonomy import label, track
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.registry import TapInfo, taps

CIFAR10 = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
BATCH = 32         # the sweep's probes_per_injection: CRASH is judged over an injection's probes
EPSILON = 0.01     # configs/inject/grid.yaml
DEMO_PROBES = 500  # probe-pool subset the demo cache covers


@dataclass
class Evaluation:
    """One injection over a probe batch, scored the way `fidnn eval` scores it."""

    idx: np.ndarray
    y: np.ndarray
    clean_logits: np.ndarray
    fault_logits: np.ndarray
    labels: np.ndarray
    tracks: np.ndarray
    taps: list[str]
    d1_clean: np.ndarray        # (n, K) per-tap scores without the fault
    d1: np.ndarray              # (n, K) per-tap scores with the fault
    d1_thresholds: np.ndarray   # (K,) D3's per-tap clean_cal thresholds
    first_alarm: np.ndarray     # (n,) tap index or -1
    d2: np.ndarray              # (n,) fused score
    taus: dict[float, float]    # α → τ_α for D2
    output_only: dict[str, np.ndarray]
    checksum: str

    def alarm(self, alpha: float) -> np.ndarray:
        return self.d2 > self.taus[alpha]


class DemoSession:
    """Holds a clean and a fault instance; the fault one is only ever mutated inside `evaluate`."""

    def __init__(self, model_id: str, clean: nn.Module, fault: nn.Module, probes_x: torch.Tensor,
                 probes_y: np.ndarray, detectors: dict, taus: dict[float, float],
                 tap_list: list[TapInfo], sat: dict[str, float], cache_dir: Path,
                 classes: tuple[str, ...] = CIFAR10, images: np.ndarray | None = None,
                 mean=None, std=None):
        self.model_id, self.clean, self.fault = model_id, clean, fault
        self.x, self.y, self.classes = probes_x, probes_y, classes
        self.images = images  # uint8 NHWC for display, aligned with probes_x
        self.mean, self.std = mean, std  # train-split normalisation, for uploaded pictures
        self.n_pool = len(probes_y)       # uploads are appended after the probe pool
        self.detectors, self.taus, self.tap_list = detectors, taus, tap_list
        self.tap_ids = [t.tap_id for t in tap_list]
        self.sat = sat
        gm = traced(clean)
        self.starts = Resume(gm).starts(stateful_modules(gm))
        cache = store.ensure(cache_dir, gm, probes_x, self.starts, tap_list, sat,
                             log=lambda _: None)
        self.runner = SuffixRunner(traced(fault), cache, tap_list, sat)
        self.clean_checksum = state_checksum(fault)
        self.targets: list[Target] = targets(fault, model_id)
        self.flips: list[dict] = []

    # --- faults ---------------------------------------------------------------------------

    def strata(self, kind: str = "bf_w") -> dict[str, list[int]]:
        return STRATA[width_of(for_mode(self.targets, kind))]

    def layers(self, kind: str = "bf_w") -> list[str]:
        return [t.layer for t in for_mode(self.targets, kind)]

    def add_flips(self, bucket: str, stratum: str, count: int, rng: np.random.Generator,
                  kind: str = "bf_w") -> list[dict]:
        """Sample like the §4.2 planner; a repeat of an applied (element, bit) is skipped."""
        if bucket not in BUCKETS:
            raise ValueError(f"bucket must be one of {BUCKETS}")
        pool = [t for t in for_mode(self.targets, kind) if t.bucket == bucket]
        new = [f | {"bucket": bucket, "stratum": stratum}
               for f in sample_flips(pool, self.strata(kind)[stratum], count, rng)]
        return self._extend(new)

    def add_flip(self, layer: str, flat_index: int, bit: int, kind: str = "bf_w") -> list[dict]:
        t = next(t for t in for_mode(self.targets, kind) if t.layer == layer)
        if not 0 <= flat_index < t.numel or not 0 <= bit < t.width:
            raise ValueError(f"{layer}: index < {t.numel} and bit < {t.width}")
        return self._extend([{"layer": t.layer, "tensor": t.tensor, "storage": t.storage,
                              "numel": t.numel, "flat_index": flat_index, "bit": bit,
                              "bucket": t.bucket, "stratum": "manual"}])

    def _extend(self, new: list[dict]) -> list[dict]:
        seen = {(f["layer"], f["tensor"], f["flat_index"], f["bit"]) for f in self.flips}
        added = [f for f in new if (f["layer"], f["tensor"], f["flat_index"], f["bit"]) not in seen]
        self.flips.extend(added)
        return added

    def reset(self) -> None:
        self.flips = []

    def describe_flip(self, f: dict) -> dict:
        """Old and new stored value of one flip, read from the clean instance."""
        mod = self.clean.get_submodule(f["layer"])
        i = int(f["flat_index"])
        if f["storage"] == "qweight":
            w = mod.weight()
            old_int = w.int_repr().contiguous().view(-1)[i].clone()  # the injector's indexing
            new_int = flip_bits_(old_int.view(1).clone(), [0], [f["bit"]])[0]
            per_ch = w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
            scale = (float(w.q_per_channel_scales()[i // w[0].numel()]) if per_ch
                     else float(w.q_scale()))
            old = float(w.dequantize().reshape(-1)[i])
            return {"old": old, "new": old + (int(new_int) - int(old_int)) * scale,
                    "old_raw": int(old_int), "new_raw": int(new_int)}
        t = mod.bias() if f["storage"] == "qbias" else getattr(mod, f["tensor"]).data
        old = t.detach().reshape(-1)[i].clone().view(1)
        new = flip_bits_(old.clone(), [0], [f["bit"]])
        return {"old": float(old[0]), "new": float(new[0])}

    # --- uploads --------------------------------------------------------------------------

    @torch.no_grad()
    def add_image(self, img: np.ndarray, label: int | None = None) -> int:
        """Append a 32×32 RGB uint8 picture; its clean pass extends the cache in memory.

        `label` defaults to the clean model's answer, since an upload has no ground truth.
        """
        if img.shape != (32, 32, 3) or img.dtype != np.uint8:
            raise ValueError("expected a 32×32×3 uint8 picture")
        if self.mean is None or self.std is None:
            raise ValueError("session has no normalisation statistics")
        mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1)
        x = (torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(255) - mean) / std
        row = store.build(traced(self.clean), x, self.starts, self.tap_list, self.sat, batch=1)
        old = self.runner.cache
        if row.tap_ids != old.tap_ids or any(row.qparams[k] != v for k, v in old.qparams.items()):
            raise RuntimeError("upload's clean pass does not match the probe cache")
        self.runner.cache = store.ProbeCache(
            {k: np.concatenate([old.cuts[k], row.cuts[k]]) for k in old.cuts}, old.qparams,
            np.concatenate([old.logits, row.logits]), old.tap_ids,
            np.concatenate([old.features, row.features]), old.key)
        self.x = torch.cat([self.x, x])
        y = int(row.logits[0].argmax()) if label is None else int(label)
        self.y = np.concatenate([self.y, np.array([y], dtype=self.y.dtype)])
        if self.images is not None:
            self.images = np.concatenate([self.images, img[None]])
        return len(self.y) - 1

    # --- evaluation -----------------------------------------------------------------------

    def batch_for(self, image: int, seed: int = 0) -> np.ndarray:
        """`image` first, then BATCH − 1 fixed other probes, so CRASH means what it does in §4.3."""
        rest = np.random.default_rng(seed).permutation(self.n_pool)
        return np.concatenate([[image], rest[rest != image][:BATCH - 1]])

    def _features(self, per_tap: dict[str, torch.Tensor], kind: type[Features]) -> Features:
        values = np.stack([per_tap[t].numpy() for t in self.tap_ids], axis=1)
        return kind(add_block_e(values, FEATURE_NAMES), self.tap_ids, FEATURE_NAMES + BLOCK_E)

    @staticmethod
    def _score(fn, feats: Features) -> np.ndarray:
        """Non-finite feature rows (a blown-up forward) score +inf: always over threshold."""
        ok = np.isfinite(feats.values).all(axis=(1, 2))
        first = fn(type(feats)(feats.values[ok], feats.taps, feats.names)) if ok.any() else None
        shape = (len(ok),) if first is None or first.ndim == 1 else (len(ok), first.shape[1])
        out = np.full(shape, np.inf)
        if first is not None:
            out[ok] = first
        return out

    @torch.no_grad()
    def evaluate(self, idx: np.ndarray) -> Evaluation:
        idx = np.asarray(idx)
        cache = self.runner.cache
        with injected_flips(self.fault, self.flips, "bf_w"):
            try:
                fault_logits, per_tap = self.runner.run({f["layer"] for f in self.flips}, idx)
            except (RuntimeError, ValueError):  # a broken forward is a CRASH, as in the sweep
                fault_logits = np.full_like(cache.logits[idx], np.nan)
                per_tap = {t: torch.full((len(idx), len(FEATURE_NAMES)), float("nan"))
                           for t in self.tap_ids}
        checksum = state_checksum(self.fault)
        if checksum != self.clean_checksum:
            raise RuntimeError("fault instance did not return to its clean state (§4.4)")

        clean_logits = np.asarray(cache.logits[idx])
        y = self.y[idx]
        labels = label(clean_logits, fault_logits, y, EPSILON, len(self.classes))
        fault = self._features(per_tap, FaultFeatures)
        clean = self._features({t: cache.tap_features(t, idx) for t in self.tap_ids}, Features)
        d1, d3 = self.detectors["D1"], self.detectors["D3_max"]
        if d1.taps != self.tap_ids:
            raise ValueError(f"detector taps {d1.taps} differ from {self.tap_ids}")
        if np.ndim(scores := self._score(d1.scores, fault)) == 1:  # every row non-finite
            scores = np.full((len(idx), len(self.tap_ids)), np.inf)
        thresholds = np.array([d3.tap_thresholds[t] for t in d1.taps])
        over = scores > thresholds
        return Evaluation(
            idx=idx, y=y, clean_logits=clean_logits, fault_logits=fault_logits, labels=labels,
            tracks=track(labels), taps=self.tap_ids,
            d1_clean=d1.scores(clean), d1=scores, d1_thresholds=thresholds,
            first_alarm=np.where(over.any(1), over.argmax(1), -1),
            d2=self._score(self.detectors["D2"].scores, fault), taus=self.taus,
            output_only=output_only_scores(np.nan_to_num(fault_logits)), checksum=checksum)

    def summary(self, ev: Evaluation, alpha: float) -> pd.DataFrame:
        """Accuracy and alarm rate, with Track S and Track H kept apart (§4.3)."""
        finite = np.isfinite(ev.fault_logits).all(1)
        pred = np.where(finite, np.nan_to_num(ev.fault_logits).argmax(1), -1)
        rows = [{"group": "all probes", "n": len(ev.y),
                 "clean_accuracy": float((ev.clean_logits.argmax(1) == ev.y).mean()),
                 "fault_accuracy": float((pred == ev.y).mean()),
                 "alarm_rate": float(ev.alarm(alpha).mean())}]
        for tr in ("S", "H"):
            m = ev.tracks == tr
            rows.append({"group": f"track {tr}", "n": int(m.sum()),
                         "alarm_rate": float(ev.alarm(alpha)[m].mean()) if m.any() else np.nan})
        return pd.DataFrame(rows)

    # --- loading --------------------------------------------------------------------------

    @classmethod
    def from_artifacts(cls, model_id: str = "m2", precision: str = "int8",
                       tap_set: str = "default", seed: int = 0,
                       art: Path = Path("artifacts"),
                       data_cfg: Path = Path("configs/data/cifar10.yaml"),
                       calib_n: int = 512) -> "DemoSession":
        from fidnn.data.loader import Batches
        from fidnn.data.prepare import load_arrays, model_classes, subset_classes
        from fidnn.models.checkpoint import load
        from fidnn.models.registry import config_path

        stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
        bundle = joblib.load(art / "detectors" / f"{stem}.joblib")
        cal = pd.read_parquet(art / "detectors" / f"{stem}_calibration.parquet")
        taus = {float(r.alpha): float(r.tau) for r in cal[cal.detector == "D2"].itertuples()}
        sat = json.loads((art / "features" / f"{stem}.json").read_text())["saturation"]["thresholds"]

        arrays = load_arrays(data_cfg, art / "data")
        subset = model_classes(config_path(model_id))
        fit_x, fit_y = subset_classes(*arrays.of("clean_fit"), subset)
        calib = [x for x, _ in Batches(fit_x[:calib_n], fit_y[:calib_n], arrays.mean,
                                       arrays.std, 64)]
        px, py = subset_classes(arrays.probe_x, arrays.probe_y, subset)
        keep = np.sort(np.random.default_rng(seed).permutation(len(py))[:DEMO_PROBES])
        x = torch.cat([xb for xb, _ in Batches(px[keep], py[keep], arrays.mean, arrays.std, 500)])
        classes = tuple(CIFAR10[c] for c in subset) if subset else CIFAR10
        return cls(model_id, load(model_id, seed, precision, art / "models", calib),
                   load(model_id, seed, precision, art / "models", calib), x, py[keep],
                   bundle["detectors"], taus, taps(model_id, tap_set), sat,
                   art / "cache" / f"demo_{stem}", classes, images=px[keep],
                   mean=arrays.mean, std=arrays.std)


def missing_prerequisites(model_id: str = "m2", precision: str = "int8",
                          tap_set: str = "default", seed: int = 0,
                          art: Path = Path("artifacts")) -> list[tuple[str, str]]:
    """(what is missing, the command that makes it), in pipeline order."""
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    need = [
        (art / "data" / "cifar10_splits.parquet", "uv run python -m fidnn data prepare"),
        (art / "models" / f"{model_id}_seed{seed}.pt",
         f"uv run python -m fidnn train {model_id} --seed {seed}"),
        (art / "features" / f"{stem}.json",
         f"uv run python -m fidnn extract {model_id} --precision {precision} --tap-set {tap_set}"),
        (art / "detectors" / f"{stem}.joblib",
         f"uv run python -m fidnn fit {model_id} --precision {precision} --tap-set {tap_set}"),
    ]
    return [(str(p), cmd) for p, cmd in need if not p.exists()]
