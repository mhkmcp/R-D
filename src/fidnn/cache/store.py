"""Clean probe-pool cache: cut-point activations, tap features and logits from one clean pass."""

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import fx

from fidnn.inject.bitflip import state_checksum
from fidnn.provenance import sidecar
from fidnn.taps.hooks import observe
from fidnn.taps.registry import TapInfo


class StaleCacheError(RuntimeError):
    """The cache on disk was built from a different model, probe pool, tap set or thresholds."""


class _Recorder(fx.Interpreter):
    """See `fx.Interpreter`; keeps the outputs of `keep` nodes and the features of tap nodes."""

    def __init__(self, gm: fx.GraphModule, keep: Sequence[str], taps: Sequence[TapInfo],
                 sat: dict[str, float]):
        super().__init__(gm)
        self.keep, self.sat = set(keep), sat
        self.tap_of = {t.path: t.tap_id for t in taps}
        self.cuts: dict[str, torch.Tensor] = {}
        self.features: dict[str, torch.Tensor] = {}

    def run_node(self, n: fx.Node):
        out = super().run_node(n)
        if n.name in self.keep:
            self.cuts[n.name] = out
        if n.op == "call_module" and n.target in self.tap_of:
            tap_id = self.tap_of[n.target]
            self.features[tap_id] = observe(out, self.sat.get(tap_id, float("inf")))
        return out


@dataclass
class ProbeCache:
    cuts: dict[str, np.ndarray]           # node name → (N, ...) int_repr (INT8) or float32
    qparams: dict[str, tuple[float, int]]  # quantised cut points only
    logits: np.ndarray                     # (N, classes)
    tap_ids: list[str]                     # graph order
    features: np.ndarray                   # (N, taps, 26)
    key: dict

    def tensor(self, name: str, idx: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(self.cuts[name][idx]))
        if name in self.qparams:
            scale, zero_point = self.qparams[name]
            return torch._make_per_tensor_quantized_tensor(t, scale, zero_point)
        return t

    def tap_features(self, tap_id: str, idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.features[idx, self.tap_ids.index(tap_id)]))


def _to_numpy(t: torch.Tensor) -> tuple[np.ndarray, tuple[float, int] | None]:
    if not t.is_quantized:
        return t.detach().numpy(), None
    if t.qscheme() != torch.per_tensor_affine:
        raise TypeError(f"cut point quantised with {t.qscheme()}; only per-tensor is cacheable")
    return t.int_repr().numpy(), (float(t.q_scale()), int(t.q_zero_point()))


def probes_hash(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()[:16]


def cache_key(gm: fx.GraphModule, x: torch.Tensor, starts: Sequence[str],
              taps: Sequence[TapInfo], sat: dict[str, float], batch: int) -> dict:
    """What a cache must agree on to be reused; JSON round-tripped so it compares to disk."""
    return json.loads(json.dumps({
        "model_checksum": state_checksum(gm), "probes": probes_hash(x), "n": len(x),
        "cuts": list(starts), "taps": [t.tap_id for t in taps],
        "sat": {t.tap_id: sat.get(t.tap_id, float("inf")) for t in taps}, "batch": batch,
    }))


@torch.no_grad()
def build(gm: fx.GraphModule, x: torch.Tensor, starts: Sequence[str], taps: Sequence[TapInfo],
          sat: dict[str, float], batch: int = 500) -> ProbeCache:
    """One clean pass over `x`. `gm` is the clean instance; it is never injected (SPEC §4.4)."""
    cuts: dict[str, list[np.ndarray]] = {s: [] for s in starts}
    qparams: dict[str, tuple[float, int]] = {}
    logits, feats, tap_ids = [], [], []
    for i in range(0, len(x), batch):
        rec = _Recorder(gm, starts, taps, sat)
        logits.append(rec.run(x[i:i + batch]).float().numpy())
        for name, t in rec.cuts.items():
            arr, qp = _to_numpy(t)
            cuts[name].append(arr)
            if qp is not None:
                qparams[name] = qp
        tap_ids = list(rec.features)
        feats.append(torch.stack([rec.features[t] for t in tap_ids], 1).numpy()
                     if tap_ids else np.zeros((len(x[i:i + batch]), 0, 0), np.float32))
    return ProbeCache({k: np.concatenate(v) for k, v in cuts.items()}, qparams,
                      np.concatenate(logits), tap_ids, np.concatenate(feats),
                      cache_key(gm, x, starts, taps, sat, batch))


def save(cache: ProbeCache, d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name, arr in cache.cuts.items():
        np.save(d / f"cut_{name}.npy", arr)
    np.save(d / "logits.npy", cache.logits)
    np.save(d / "features.npy", cache.features)
    record = {"key": cache.key, "tap_ids": cache.tap_ids,
              "qparams": {k: list(v) for k, v in cache.qparams.items()},
              **sidecar(cache.key, seed=-1, device="cpu")}
    (d / "cache.json").write_text(json.dumps(record, indent=2, default=str))


def load(d: Path, key: dict) -> ProbeCache:
    """Memory-mapped; refuses a cache built for anything other than `key`."""
    meta_path = d / "cache.json"
    if not meta_path.exists():
        raise StaleCacheError(f"no cache at {d}")
    meta = json.loads(meta_path.read_text())
    if meta["key"] != key:
        diff = sorted(k for k in key if meta["key"].get(k) != key[k])
        raise StaleCacheError(f"cache at {d} differs in {diff}")
    return ProbeCache({c: np.load(d / f"cut_{c}.npy", mmap_mode="r") for c in key["cuts"]},
                      {k: (float(v[0]), int(v[1])) for k, v in meta["qparams"].items()},
                      np.load(d / "logits.npy"), meta["tap_ids"],
                      np.load(d / "features.npy", mmap_mode="r"), meta["key"])


def ensure(d: Path, gm: fx.GraphModule, x: torch.Tensor, starts: Sequence[str],
           taps: Sequence[TapInfo], sat: dict[str, float], batch: int = 500,
           log=print) -> ProbeCache:
    """Load the cache at `d`, or rebuild it from the clean instance when it is missing or stale."""
    key = cache_key(gm, x, starts, taps, sat, batch)
    try:
        return load(d, key)
    except StaleCacheError as exc:
        log(f"building probe cache: {exc}")
    save(build(gm, x, starts, taps, sat, batch), d)
    return load(d, key)
